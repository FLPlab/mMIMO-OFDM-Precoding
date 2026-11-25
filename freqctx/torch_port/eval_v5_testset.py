"""Notre precodeur (32 bits et quantifie ART, entree normalisee) sur le jeu de test COMPLET de BitAdapt v5
(St-Catherine, chargeur Pipeline.utils.Data_Load du depot BitAdaptHLS, comme eval_baselines_full.py),
pour comparer aux debits du front v5 sur exactement les memes canaux. Controle : RZF et notre WMMSE
(I = 1, 5, 10, 25, sur les 20 000 premiers canaux comme la v5) doivent redonner les baselines v5
(RZF 17,14 ; WMMSE 19,29 / 21,53 / 22,56 / 23,86).

    cd <BitAdaptHLS> && python -m torch_port.eval_v5_testset     (PYTHONPATH = freqctx)
"""
import json
import os
import sys

import torch

FREQCTX = "/users/sala/Documents/test_projet/Trans/freq/Sionna/freqctx"
V5 = "/export/tmp/sala/home_archive/ELE6310/ELE6310E"
sys.path[:0] = [FREQCTX, V5]
from torch_port.art_stecath import quantize, quantize_products  # noqa: E402
from torch_port.models import PairPrecoder  # noqa: E402
from torch_port.stecath import DEV, OUT_W, RESULTS, lab_rate, ours_input, rzf, wmmse  # noqa: E402
from Pipeline.input_args import input_args  # noqa: E402
from Pipeline.utils import utils_  # noqa: E402


def loader(bs=2000):
    sys.argv = [sys.argv[0]]
    a = input_args().args
    a.batch_size, a.scenario = bs, ["stecath"]
    return utils_(a).Data_Load()[1]


def main():
    fp32 = PairPrecoder(d=16, hidden=32, layers=8, inp="norm").to(DEV)
    fp32.load_state_dict(torch.load(f"{OUT_W}/pair_h32_norm.pt", map_location=DEV, weights_only=True))
    qnet = PairPrecoder(d=16, hidden=32, layers=8, inp="norm")
    quantize(qnet)
    quantize_products(qnet)
    sd = torch.load(f"{OUT_W}/art_pair_h32_lam5_qprod_norm.pt", map_location="cpu", weights_only=True)
    ref = qnet.state_dict()
    qnet.load_state_dict({k: v.reshape(ref[k].shape) for k, v in sd.items()})   # echelles : [] -> [1]
    qnet = qnet.to(DEV)
    fp32.eval(), qnet.eval()
    small = {}
    for tag, hid, nl in (("art_pair_h32_L4_lam5_qprod_norm", 32, 4), ("art_pair_h16_L4_lam5_qprod_norm", 16, 4),
                         ("art_pair_h16_L2_lam5_qprod_norm", 16, 2)):
        m = PairPrecoder(d=16, hidden=hid, layers=nl, inp="norm")
        quantize(m)
        quantize_products(m)
        sd = torch.load(f"{OUT_W}/{tag}.pt", map_location="cpu", weights_only=True)
        rf = m.state_dict()
        m.load_state_dict({k: v.reshape(rf[k].shape) for k, v in sd.items()})
        small[tag] = m.to(DEV).eval()
    s = {k: [] for k in ("notre precodeur 32 bits", "notre precodeur quantifie", "RZF",
                         "WMMSE I=1", "WMMSE I=5", "WMMSE I=10", "WMMSE I=25", *small)}
    n = 0
    with torch.no_grad():
        for Hr, _ in loader():
            H = Hr.squeeze(2).permute(0, 2, 1).to(torch.complex64)        # reconstruct_H v5 : (B, N_T, N_U)
            H = H.transpose(1, 2).contiguous().to(DEV)                    # notre convention : (B, K, M)
            s["notre precodeur 32 bits"].append(lab_rate(fp32(*ours_input(H)), H).cpu())
            s["notre precodeur quantifie"].append(lab_rate(qnet(*ours_input(H)), H).cpu())
            s["RZF"].append(lab_rate(rzf(H), H).cpu())
            for tag, m in small.items():
                s[tag].append(lab_rate(m(*ours_input(H)), H).cpu())
            if n < 20000:
                for i in (1, 5, 10, 25):
                    s[f"WMMSE I={i}"].append(lab_rate(wmmse(H, i), H).cpu())
            n += H.shape[0]
    out = {k: {"sr": float(torch.cat(v).mean()), "n": int(torch.cat(v).numel())} for k, v in s.items()}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(RESULTS, "v5_testset_ours.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
