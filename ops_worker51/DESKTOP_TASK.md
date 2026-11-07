# Claude Desktop local scheduled task — exact setup (owner does this in the Desktop UI; CLI cannot register it)

Name: `sanya-nuclear-gpu-worker` · Local / Only on this computer · Working folder: `C:\nuclear-doe-operator\b-sampler`
Permission mode: **Ask permissions** (default) — then use Run now and "always allow" the safe prompts (see 4).

## Instructions (paste into the task's instructions box)
You are the scheduled #51 GPU worker on DESKTOP-T1K698V (RTX 4070 Ti), GitHub user MVRto01, repo
NinjaMoons/nuclear-doe-2026. Run exactly:
`C:\nuclear-doe-operator\desktop-41\.venv-41\Scripts\python.exe -I C:\nuclear-doe-operator\ops\worker51\precheck.py`
If action is NO_TASK, LOCKED, BUSY or GH_ERROR: reply with one line and stop — no other tool calls.
If this is the first run of this task (no line `DESKTOP_TASK_TESTED` in C:\nuclear-doe-operator\ops\worker51\state.json
notes), ALSO post one comment to issue #51 starting `<!-- sanya-desktop-task-test -->` with
`SANYA_DESKTOP_TASK_TEST=PASS | HOST=DESKTOP-T1K698V | TIMESTAMP_MSK | precheck action | GIT_SHA of C:\nuclear-doe-operator\b-sampler | session = new Desktop scheduled session`
and then append "DESKTOP_TASK_TESTED <time>" to state.json "notes" (list).
If action is TASK: follow C:\nuclear-doe-operator\ops\worker51\WORKER.md exactly (card fields, rights, science GO,
`phase` gate, claim, one GPU job in its own worktree, publish to #51, release tick.lock). Never modify V1/V2/4ch
packages or checkpoints, never decode TEST, never submit officially, no paid actions, no public ports.

## Steps (Claude Desktop >= 1.1.5368)
1. Claude Desktop → **Code** tab → sidebar **Scheduled** (or **Routines**) → **New task** (if asked: **Set up manually**).
2. If **Only on this computer** is at the top of the dialog, turn it on first. Otherwise: Name → instructions →
   **Advanced settings → General → Only on this computer** = on → **Local folder** = `C:\nuclear-doe-operator\b-sampler`.
3. Trigger: **Hourly** (the form's minimum). Click **Create**, trust the folder if asked.
   For 12-15 min: open a normal Desktop Code session in the same folder and type
   "set my scheduled task sanya-nuclear-gpu-worker to run every 15 minutes" (Claude may set < 1 h; the form cannot).
4. On the task page click **Run now**. Open the run from **History**, answer each permission prompt with
   **always allow** (python precheck, gh, git, nvidia-smi). Do not enable a full bypass mode.
5. Confirm: a new comment `SANYA_DESKTOP_TASK_TEST=PASS` in #51 + a row in **History**.
6. Settings → Desktop app (or This computer → System) → **Keep computer awake** = on. Leave Claude Desktop open.
7. When step 5 is confirmed, stop the old in-session CLI cron (tell the CLI Claude "CronDelete 5752bdd8" or close
   that terminal) so only ONE worker polls #51 (precheck also has a 50-min tick.lock guard for TASK runs).
