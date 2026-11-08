#!/usr/bin/env bash
# R25 PHASE-2 PRIMARY paired pilot (TRAIN512, seed 2026, 6000 updates/arm, one endpoint, VAL16 once per arm).
# Run ONLY after explicit independent SCIENCE_GATE=GO for NIH-POST-R25-GF-DDFN-037. Host: Git Bash (MINGW64) on Windows.
# Fail-fast: any error -> RUN_FAILED + nonzero exit. Existing output dirs are NEVER reused (python mkdir exist_ok=False).
# Order: train control -> train dpsi -> verify both (summary, ckpt sha, same V2, same seed/updates) -> VAL eval both -> compare.
set -euo pipefail
trap 'echo "RUN_FAILED line $LINENO"; exit 1' ERR
OUT=${R25_OUT:-C:/ndoe-out/r25_p2}
UPD=${R25_UPDATES:-6000}      # pipeline dry test only: R25_UPDATES=20 R25_SKIP_VAL=1
SKIP_VAL=${R25_SKIP_VAL:-0}
R25=/c/nuclear-doe-operator/r25/research/post_submitted/r25_gf_ddfn
cd /c/nuclear-doe-operator/r22
export PYTHONPATH="C:/nuclear-doe-operator/r22;C:/nuclear-doe-operator/r25/research/post_submitted"
P=/c/nuclear-doe-operator/desktop-41/.venv-41/Scripts/python.exe
V2=C:/nuclear-doe-operator/desktop-41/cand/stem_native_s0_step_012000_gpu41.pt
M=C:/nuclear-doe-operator/desktop-41/corpus/manifest_expanded.json
uname -s | grep -q MINGW || { echo "RUN_FAILED not Git Bash"; exit 1; }
if [ -e "$OUT" ]; then echo "RUN_FAILED $OUT exists (no reuse of partial outputs)"; exit 1; fi
[ "$(sha256sum "$V2" | cut -c1-64)" = 19b05dad34601a8b5ea4c78e7cd6904b0c639a218f8fbcdeabc2afbb5d6f083f ] || { echo "RUN_FAILED V2 sha"; exit 1; }
git -C /c/nuclear-doe-operator/r25 diff --quiet HEAD -- research/post_submitted/r25_gf_ddfn || { echo "RUN_FAILED uncommitted source"; exit 1; }
if nvidia-smi --query-compute-apps=name --format=csv,noheader | grep -qi python; then echo "RUN_FAILED GPU busy"; exit 1; fi
mkdir -p "$OUT"
SRC=$(git -C /c/nuclear-doe-operator/r25 rev-parse HEAD); echo "SOURCE $SRC" | tee "$OUT/source.txt"
(cd "$R25" && sha256sum *.py) > "$OUT/source_sha256.txt"
for arm in control dpsi; do
  echo "TRAIN $arm start $(date -Is)"
  "$P" -m r25_gf_ddfn.run_r25_notest --arm $arm --seed 2026 --updates "$UPD" --device cuda --no-eval \
       --v2 "$V2" --manifest "$M" --out "$OUT/$arm" > "$OUT/$arm.log" 2>&1
  tail -1 "$OUT/$arm.log"
done
"$P" - "$OUT" "$UPD" <<'EOF'
import hashlib, json, sys
from pathlib import Path
o = Path(sys.argv[1]); s = {a: json.load(open(o / a / "r25_summary.json")) for a in ("control", "dpsi")}
for a, v in s.items():
    ck = Path(v["ckpt"]); assert ck.stat().st_size > 0 and hashlib.sha256(ck.read_bytes()).hexdigest() == v["ckpt_sha256"], a
    assert v["updates"] == int(sys.argv[2]) and v["seed"] == 2026 and v["train_pool"] == "all512" and v["reload_max_abs_diff"] == 0.0, a
assert s["control"]["v2_sha256"] == s["dpsi"]["v2_sha256"] and s["control"]["v2_sha256"].startswith("19b05dad")
assert s["control"]["script_sha256"] == s["dpsi"]["script_sha256"]
(o / "ckpts.txt").write_text("\n".join(f'{a} {v["ckpt"]} {v["ckpt_sha256"]}' for a, v in s.items()) + "\n")
print("PROVENANCE_OK")
EOF
if [ "$SKIP_VAL" = 1 ]; then echo "DRY_TEST_DONE (VAL skipped)"; exit 0; fi
while read -r arm ck sha; do
  "$P" -m r25_gf_ddfn.eval_r25 --ckpt "$ck" --expect-sha "$sha" --manifest "$M" --out "$OUT/eval_$arm" --device cuda > "$OUT/eval_$arm.log" 2>&1
  tail -1 "$OUT/eval_$arm.log"
done < "$OUT/ckpts.txt"
E2=$(ls -d /c/ndoe-out/r25_audit/val16/export/*/val)
"$P" -m r25_gf_ddfn.compare_r25 "$OUT/compare" v2="$E2" control="$OUT/eval_control/export/step_$(printf %06d "$UPD").pt/val" \
     treatment="$OUT/eval_dpsi/export/step_$(printf %06d "$UPD").pt/val" adam_oracle_DIAGNOSTIC=C:/nuclear-doe-operator/desktop-41/b75/oracle/O2_adam_r3_from_stem \
     > "$OUT/compare.log" 2>&1
echo "ALL_DONE $(date -Is)"
