#!/bin/bash
# Attend le modele a pilote aleatoire (srvgrm08), rapatrie ses poids, puis
# le passe dans le test de robustesse sur srvgrm12.
cd /users/sala/Documents/test_projet/Trans/freq/Sionna/freqctx
until [ -f results/phase7/pilot_rand.json ]; do sleep 120; done
rsync -a srvgrm08:/export/tmp/sala/freqctx/weights/phase7/pilot_rand.weights.h5 /export/tmp/sala/freqctx/weights/phase7/
source /export/tmp/sala/miniconda/bin/activate tf-gpu
export LD_LIBRARY_PATH=/export/tmp/sala/miniconda/envs/tf-gpu/lib:$LD_LIBRARY_PATH
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 python -u -m experiments.phase8_robust \
    --models "modele (pilote aleatoire)" --out robustness_pilot_rand
