#!/bin/bash
# R25 PHASE-2 PRIMARY paired pilot (TRAIN512, seed 2026, 6000 updates/arm, one endpoint, VAL16 once).
# Run ONLY after explicit independent SCIENCE_GATE=GO for NIH-POST-R25-GF-DDFN-037. Skips existing outputs. One GPU job at a time.
cd /c/nuclear-doe-operator/r22
export PYTHONPATH="C:/nuclear-doe-operator/r22;C:/nuclear-doe-operator/r25/research/post_submitted"
P=/c/nuclear-doe-operator/desktop-41/.venv-41/Scripts/python.exe
V2=C:/nuclear-doe-operator/desktop-41/cand/stem_native_s0_step_012000_gpu41.pt
M=C:/nuclear-doe-operator/desktop-41/corpus/manifest_expanded.json
for arm in control dpsi; do o=C:/ndoe-out/r25_p2_$arm
  [ -e $o ] && { echo "exists $o"; continue; }
  $P -m r25_gf_ddfn.run_r25_notest --arm $arm --seed 2026 --updates 6000 --device cuda --v2 $V2 --manifest $M --out $o > $o.log 2>&1 || echo "FAIL $o"
  tail -2 $o.log
done
E2=$(ls -d /c/ndoe-out/r25_audit/val16/export/*/val)
$P -m r25_gf_ddfn.compare_r25 C:/ndoe-out/r25_p2_compare v2=$E2 control=C:/ndoe-out/r25_p2_control/eval_val/export/step_006000.pt/val treatment=C:/ndoe-out/r25_p2_dpsi/eval_val/export/step_006000.pt/val > /c/ndoe-out/r25_p2_compare.log 2>&1 || echo "FAIL compare"
echo ALL_DONE
