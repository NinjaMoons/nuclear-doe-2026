#!/usr/bin/env python3
"""Single-call Gemini CLI research bridge for Claude Code, with no automatic retries.

Run only with trusted, sanitized research prompts; keeps Gemini cwd outside the repo.
Does not install Gemini, sign in, bypass quotas, or start unattended monitoring.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", type=Path, required=True, help="UTF-8 text prompt file")
    p.add_argument("--out", type=Path, required=True, help="Local ignored output directory")
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()

    if not 10 <= args.timeout <= 600:
        p.error("--timeout must be 10..600 seconds")
    source = args.prompt.resolve()
    if not source.is_file():
        p.error("prompt file does not exist")
    prompt = source.read_text(encoding="utf-8")
    if not prompt.strip() or len(prompt) > 30000:
        p.error("prompt must be between 1 and 30,000 characters")
    executable = shutil.which("gemini")
    if not executable:
        print("GEMINI_NOT_INSTALLED: no request sent; continue with Opus", file=sys.stderr)
        return 2

    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    suffix = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + suffix[:10]
    run_dir = destination / run_id
    run_dir.mkdir(exist_ok=False)
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    receipt = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prompt_sha256": suffix,
        "source": "gemini-cli-headless",
        "status": "STARTED",
    }
    try:
        # Isolate the Gemini agent working directory so tools do not modify the repo.
        with tempfile.TemporaryDirectory(prefix="gemini-research-") as sandbox:
            result = subprocess.run(
                [executable, "--prompt", prompt, "--output-format", "json"],
                cwd=sandbox,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout,
                check=False,
            )
        (run_dir / "raw_stdout.json").write_text(result.stdout, encoding="utf-8")
        # Keep stderr LOCAL only; may contain environment-specific diagnostic info.
        (run_dir / "local_stderr.txt").write_text(result.stderr, encoding="utf-8")
        receipt["exit_code"] = result.returncode
        if result.returncode:
            error = (result.stderr + "\n" + result.stdout).lower()
            receipt["status"] = ("QUOTA_OR_RATE_LIMIT" if
                any(term in error for term in ("quota", "429", "rate limit", "exhausted"))
                else "CLI_ERROR")
        else:
            try:
                payload = json.loads(result.stdout)
                reply = payload.get("response")
                if not isinstance(reply, str):
                    raise ValueError("Gemini returned no response text")
                (run_dir / "reply.md").write_text(reply, encoding="utf-8")
                receipt["stats"] = payload.get("stats")
                receipt["status"] = "OK"
            except (ValueError, TypeError) as exc:
                receipt["status"] = "PARSE_ERROR"
                receipt["detail"] = str(exc)
    except subprocess.TimeoutExpired:
        receipt["status"] = "TIMEOUT"
    finally:
        receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (run_dir / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"{receipt['status']} {run_dir}")
    return 0 if receipt["status"] == "OK" else 3


if __name__ == "__main__":
    raise SystemExit(main())
