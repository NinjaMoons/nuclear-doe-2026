"""#51 worker pre-check (deterministic, no AI). Prints one JSON line:
{"action": "NO_TASK"|"TASK"|"BUSY", ...}. Reads new #51 comments after state.last_comment_id, finds task cards with
STATUS=READY addressed to MVRto01/DESKTOP-T1K698V whose TASK_ID is not in the ledger; reports GPU python jobs.
Only advances last_comment_id when there is no unprocessed READY card (so a card is never skipped)."""
import json, re, subprocess, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

GH = r"C:\Program Files\GitHub CLI\gh.exe"
REPO = "NinjaMoons/nuclear-doe-2026"
STATE = Path(__file__).with_name("state.json")
MSK = timezone(timedelta(hours=3))


def load():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"last_comment_id": 0, "processed": {}, "ticks": 0}


LOCK = Path(__file__).with_name("tick.lock")


def phase(now):
    """Team gates MSK (coordinator #51/6714140160, 8 Nov): returns the allowed work class."""
    hm = now.hour * 60 + now.minute
    if now.hour >= 12:
        return "pre-0200 (old night)"
    if hm < 4 * 60 + 30:
        return "TRAIN_ALLOWED_WITH_GO"          # 02:00-04:30 bounded preregistered TRAIN-only training
    if hm < 5 * 60 + 15:
        return "FINISH_ONLY"                    # 04:30-05:15 finish in-flight runs, export, compare
    if hm < 6 * 60:
        return "ACCEPTANCE_ONLY"                # 05:15-06:00 independent acceptance / packaging
    return "FROZEN"                             # after 06:00 no model changes


def main():
    import os, time
    now0 = time.time()
    if LOCK.exists() and now0 - LOCK.stat().st_mtime < 50 * 60:
        print(json.dumps({"action": "LOCKED", "note": "another worker tick/task holds tick.lock (<50 min old)"})); return
    s = load()
    s["ticks"] = s.get("ticks", 0) + 1
    now = datetime.now(MSK)
    out = subprocess.run([GH, "api", f"repos/{REPO}/issues/51/comments?per_page=100", "--paginate"],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode:
        print(json.dumps({"action": "GH_ERROR", "stderr": out.stderr[-300:]})); return
    comments = []
    for chunk in re.findall(r"\[.*?\](?=\[|\s*$)", out.stdout.strip(), re.S) or [out.stdout]:
        comments += json.loads(chunk)
    new = [c for c in comments if c["id"] > s["last_comment_id"]]
    tasks = []
    for c in new:
        b = c["body"]
        if c["user"]["login"] not in ("NinjaMoons", "ggmaster913"):   # task cards come from coordinator/science gate only
            continue
        m = re.search(r"TASK_ID\s*[=:]\s*`?([A-Za-z0-9_.\-]+)", b)
        if not m or not re.search(r"STATUS\s*=\s*READY", b) or not re.search(r"(TO\s*=\s*@?|for\s+@?)(MVRto01|DESKTOP-T1K698V)|MVRto01\s*/\s*DESKTOP-T1K698V", b):
            continue
        tid = m.group(1)
        other_hosts = set(re.findall(r"DESKTOP-[A-Z0-9]{5,}", b)) - {"DESKTOP-T1K698V"}
        to = re.search(r"TO\s*=\s*\S+", b)
        if to and "DESKTOP-T1K698V" not in to.group(0) and re.search(r"DESKTOP-[A-Z0-9]{5,}", to.group(0)):
            continue                      # explicitly addressed to another physical host (e.g. the laptop)
        if other_hosts and "DESKTOP-T1K698V" not in b:
            continue
        if tid in s["processed"]:
            continue
        tasks.append({"task_id": tid, "comment_id": c["id"], "author": c["user"]["login"], "url": c["html_url"],
                      "created_at": c["created_at"]})
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    gpu_py = [l.strip() for l in gpu.splitlines() if "python" in l.lower()]
    unsure = [c["id"] for c in new if re.search(r"TASK_ID\s*[=:]", c["body"]) and re.search(r"STATUS\s*=\s*READY", c["body"])
              and c["user"]["login"] in ("NinjaMoons", "ggmaster913") and not any(t["comment_id"] == c["id"] for t in tasks)]
    if not tasks and new:
        s["last_comment_id"] = max(c["id"] for c in new)
    if unsure:
        s.setdefault("needs_human_look", sorted(set(s.get("needs_human_look", []) + unsure)))
    s["last_tick_msk"] = now.isoformat(timespec="seconds")
    STATE.write_text(json.dumps(s, indent=1), encoding="utf-8")
    action = "BUSY" if (tasks and gpu_py) else ("TASK" if tasks else "NO_TASK")
    if action == "TASK":
        LOCK.write_text(now.isoformat(timespec="seconds"), encoding="utf-8")   # released by the task handler
    print(json.dumps({"action": action, "now_msk": now.strftime("%H:%M"), "new_comments": len(new), "tasks": tasks,
                      "gpu_python": gpu_py, "phase": phase(now), "ticks": s["ticks"]}))


if __name__ == "__main__":
    main()
