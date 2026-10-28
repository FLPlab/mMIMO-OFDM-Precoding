#!/bin/bash
# File d'attente phase 7, etape 1 de l'interpolation : chaque run part quand
# le run de la phase 7 qui occupait son GPU a fini (fichier de resultat present).
cd /users/sala/Documents/test_projet/Trans/freq/Sionna/freqctx
R=results/phase7
wait_for() { until [ -f "$R/$1" ]; do sleep 60; done; }
(wait_for separate.json;  bash scripts/run7.sh 0 interp_s3 --variant interp --step 3) &
(wait_for lmmse.json;     bash scripts/run7.sh 1 interp_s4 --variant interp --step 4) &
(wait_for sub_17rb.json;  bash scripts/run7.sh 1 interp_s6 --variant interp --step 6) &
wait
