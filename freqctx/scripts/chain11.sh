#!/bin/bash
# Phase 11 : evaluation a puissances reelles des que les trois entrainements ont fini.
cd /users/sala/Documents/test_projet/Trans/freq/Sionna/freqctx
R=results/phase7
until [ -f $R/pilot_rand_ph32_norm.json ] && [ -f $R/pilot_rand_ph32_norm_relpow.json ] && [ -f $R/pilot_rand_ph32_relpow.json ]; do sleep 60; done
source /export/tmp/sala/miniconda/bin/activate tf-gpu
export LD_LIBRARY_PATH=/export/tmp/sala/miniconda/envs/tf-gpu/lib:$LD_LIBRARY_PATH
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 python -u -m experiments.phase11_power
