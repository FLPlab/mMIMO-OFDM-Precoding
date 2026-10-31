#!/bin/bash
# Phase 9, reglage de taille : les deux derniers tests partent quand les verifications
# de symetrie (banc phase 6) liberent les GPU 0 et 1 de srvgrm12.
cd /users/sala/Documents/test_projet/Trans/freq/Sionna/freqctx
R=results/phase6
(until [ -f $R/trig_L8_d16_h64_untied_sym.json ]; do sleep 60; done; bash scripts/run7.sh 0 size_ed16 --variant pilot_rand --est_d 16) &
(until [ -f $R/trig_L8_d16_h128_untied_sym.json ]; do sleep 60; done; bash scripts/run7.sh 1 size_el1 --variant pilot_rand --est_layers 1) &
wait
