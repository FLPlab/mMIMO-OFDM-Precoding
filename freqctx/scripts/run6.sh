#!/bin/bash
# Lance un run de phase 6 en arriere-plan, sortie non tamponnee.
# usage : scripts_run6.sh <gpu> <log_name> <args phase6_train...>
cd /users/sala/Documents/test_projet/Trans/freq/Sionna/freqctx
if [ -d /export/tmp/sala/anaconda3/envs/freqctx ]; then
  source /export/tmp/sala/anaconda3/bin/activate freqctx; LIB=/export/tmp/sala/anaconda3/envs/freqctx/lib
else
  source /export/tmp/sala/miniconda/bin/activate tf-gpu; LIB=/export/tmp/sala/miniconda/envs/tf-gpu/lib
fi
export LD_LIBRARY_PATH=$LIB:$LD_LIBRARY_PATH
gpu=$1; log=$2; shift 2
CUDA_VISIBLE_DEVICES=$gpu nohup python -u -m experiments.phase6_train "$@" > /export/tmp/sala/freqctx/logs/p6_$log.log 2>&1 < /dev/null &
echo "lance $log sur GPU $gpu (pid $!)"
