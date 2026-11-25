#!/bin/bash
# Orchestrateur de synthese HLS (csynth seulement) pour le modele de cout FPGA.
# A executer SUR srvgrm07 : ssh -o BatchMode=yes srvgrm07 'bash /users/sala/Documents/test_projet/hls_work/cost_model/run_all.sh'
# Au plus 4 synthese en parallele (licence partagee avec un autre agent).
set -u

SHARED=/users/sala/Documents/test_projet/hls_work/cost_model
LOCAL=/export/tmp/sala/hls_work/cost_model
MAX_PAR=4
LICENSE="2100@srvgrm07.grm.polymtl.ca:6062@cadpass.grm.polymtl.ca"
SETTINGS=/CMC/tools/xilinx/Vitis_2022.2/Vitis_HLS/2022.2/settings64.sh

mkdir -p "$LOCAL"
STATUS_DIR="$SHARED/_status"
mkdir -p "$STATUS_DIR"

run_one() {
    local point="$1"
    local src="$SHARED/$point"
    local dst="$LOCAL/$point"
    local marker="$STATUS_DIR/${point}.done"
    local failmarker="$STATUS_DIR/${point}.fail"
    rm -f "$marker" "$failmarker"

    rm -rf "$dst"
    mkdir -p "$dst"
    cp "$src"/dense_kernel.cpp "$src"/run_synth.tcl "$dst"/

    (
        export XILINXD_LICENSE_FILE="$LICENSE"
        source "$SETTINGS"
        cd "$dst" || exit 1
        START=$(date +%s)
        vitis_hls -f run_synth.tcl > synth.log 2>&1
        RC=$?
        END=$(date +%s)
        echo "DURATION=$((END-START))s RC=$RC" >> synth.log
    )

    # recopie du rapport (et du log) sur le disque partage, meme en cas d'echec
    mkdir -p "$src/report"
    if [ -f "$dst/proj/solution1/syn/report/csynth.xml" ]; then
        cp "$dst/proj/solution1/syn/report/csynth.xml" "$src/report/"
        cp "$dst/proj/solution1/syn/report/dense_kernel_csynth.rpt" "$src/report/" 2>/dev/null
        touch "$marker"
    else
        touch "$failmarker"
    fi
    cp "$dst/synth.log" "$src/report/synth.log" 2>/dev/null
}

export -f run_one
export SHARED LOCAL LICENSE SETTINGS STATUS_DIR

POINTS=$(cd "$SHARED" && ls -d case*/ | sed 's#/##')
TOTAL=$(echo "$POINTS" | wc -w)
echo "[run_all] $TOTAL points a synthetiser, max $MAX_PAR en parallele"
echo "$POINTS" | tr ' ' '\n' > "$STATUS_DIR/all_points.txt"

njobs=0
for point in $POINTS; do
    run_one "$point" &
    njobs=$((njobs+1))
    if [ "$njobs" -ge "$MAX_PAR" ]; then
        wait -n
        njobs=$((njobs-1))
    fi
done
wait

NDONE=$(ls "$STATUS_DIR"/*.done 2>/dev/null | wc -l)
NFAIL=$(ls "$STATUS_DIR"/*.fail 2>/dev/null | wc -l)
echo "[run_all] termine : $NDONE OK, $NFAIL echecs sur $TOTAL"
