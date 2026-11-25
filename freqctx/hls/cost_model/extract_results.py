#!/usr/bin/env python3
"""
Extrait LUT/FF/DSP/BRAM/latence/II/periode-estimee a partir des rapports
csynth.xml et produit cost_table.csv / cost_table.json.

A executer en local (disque partage) une fois run_all.sh termine :
  /export/tmp/sala/miniconda/envs/bitadapt/bin/python extract_results.py
"""
import os
import json
import csv
import re
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_point_name(name):
    m = re.match(r"case([AB])_bw(\d+)_ba(\d+)(?:_(lut|dsp))?$", name)
    if not m:
        return None
    case, bw, ba, bindop = m.groups()
    return {
        "case": case,
        "bw": int(bw),
        "ba": int(ba),
        "bindop": bindop if bindop else ("const" if case == "A" else "default"),
    }


def parse_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    def findtext(tag_path, default=None):
        el = root.find(tag_path)
        return el.text if el is not None else default

    res = {}
    res["clock_target_ns"] = float(findtext(".//UserAssignments/TargetClockPeriod", "nan"))
    res["clock_estimated_ns"] = float(findtext(".//SummaryOfTimingAnalysis/EstimatedClockPeriod", "nan"))
    res["latency_best"] = findtext(".//SummaryOfOverallLatency/Best-caseLatency")
    res["latency_worst"] = findtext(".//SummaryOfOverallLatency/Worst-caseLatency")
    res["ii_min"] = findtext(".//SummaryOfOverallLatency/Interval-min")
    res["ii_max"] = findtext(".//SummaryOfOverallLatency/Interval-max")
    res["pipeline_depth"] = findtext(".//SummaryOfOverallLatency/PipelineDepth")

    area = root.find(".//AreaEstimates/Resources")
    if area is not None:
        for tag in ["LUT", "FF", "DSP", "BRAM_18K", "URAM"]:
            el = area.find(tag)
            res[tag] = int(el.text) if el is not None and el.text is not None else None

    try:
        clk = float(res["clock_estimated_ns"])
        res["fmax_mhz"] = round(1000.0 / clk, 1) if clk > 0 else None
    except (TypeError, ValueError):
        res["fmax_mhz"] = None

    return res


def main():
    points = sorted(d for d in os.listdir(ROOT)
                     if d.startswith("case") and os.path.isdir(os.path.join(ROOT, d)))

    rows = []
    missing = []
    for p in points:
        meta = parse_point_name(p)
        if meta is None:
            continue
        xml_path = os.path.join(ROOT, p, "report", "csynth.xml")
        if not os.path.exists(xml_path):
            missing.append(p)
            continue
        try:
            res = parse_xml(xml_path)
        except Exception as e:
            missing.append(f"{p} (parse error: {e})")
            continue

        n_mac = 16 * 16  # IN_N * OUT_N
        row = {
            "point": p,
            "case": meta["case"],
            "bw": meta["bw"],
            "ba": meta["ba"],
            "bindop": meta["bindop"],
            "n_mac": n_mac,
            **res,
        }
        if row.get("LUT") is not None:
            row["lut_per_mac"] = row["LUT"] / n_mac
        if row.get("DSP") is not None:
            row["dsp_per_mac"] = row["DSP"] / n_mac
        if row.get("FF") is not None:
            row["ff_per_mac"] = row["FF"] / n_mac
        rows.append(row)

    rows.sort(key=lambda r: (r["case"], r["bindop"], r["bw"], r["ba"]))

    csv_path = os.path.join(ROOT, "cost_table.csv")
    json_path = os.path.join(ROOT, "cost_table.json")

    fieldnames = ["point", "case", "bindop", "bw", "ba", "n_mac",
                  "LUT", "FF", "DSP", "BRAM_18K", "URAM",
                  "lut_per_mac", "dsp_per_mac", "ff_per_mac",
                  "latency_best", "latency_worst", "ii_min", "ii_max",
                  "pipeline_depth", "clock_target_ns", "clock_estimated_ns", "fmax_mhz"]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"{len(rows)} points extraits -> {csv_path}, {json_path}")
    if missing:
        print(f"{len(missing)} points manquants/en erreur :")
        for m in missing:
            print("  ", m)


if __name__ == "__main__":
    main()
