# research_lambda_wrapper.py — RESEARCH ONLY, не для #8 (по образцу @playerr456, #6/6705711952)
import hashlib, json, sys
from pathlib import Path
import doe_models.train_multiplane as tm
lam, argv = float(sys.argv[1]), sys.argv[2:]
assert tm.LSQ_MIX_OBJECTIVE == "lsq_gain_nmse_plus_0.1_contract_mse_v1" and tm._LSQ_CONTRACT_WEIGHT == 0.1
tm._LSQ_CONTRACT_WEIGHT = lam
rc = tm.main(argv)
out = Path(argv[argv.index("--out-dir") + 1])
(out / "research_lambda.json").write_text(json.dumps({
    "lambda": lam, "objective_label_in_logs_is_stale": lam != 0.1,
    "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}), encoding="utf-8")
sys.exit(rc)
