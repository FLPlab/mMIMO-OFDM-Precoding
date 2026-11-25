#!/bin/bash
# Phase 13 -- modeles plus petits de notre precodeur sur St-Catherine (32 bits), puis ART lambda = 5 (un seul).
cd /users/sala/Documents/test_projet/Trans/freq/Sionna/freqctx
PY=/export/tmp/sala/miniconda/envs/bitadapt/bin/python
L=/export/tmp/sala/freqctx/logs
W=/export/tmp/sala/freqctx/weights/stecath
export CUDA_VISIBLE_DEVICES=${1:-0}
until [ -f $W/pair_h16_L4_norm.pt ]; do sleep 30; done            # run 32 bits deja en cours
$PY -u -m torch_port.stecath --train --inp norm --layers 2 --hidden 16 > $L/p13_fp32_L2_h16.log 2>&1
for cfg in "4 32" "4 16" "2 16"; do set -- $cfg
  $PY -u -m torch_port.art_stecath --qprod --inp norm --layers $1 --hidden $2 --lam 5 > $L/p13_art_L$1_h$2_lam5.log 2>&1
done
echo FINI > $L/p13_pareto_done
