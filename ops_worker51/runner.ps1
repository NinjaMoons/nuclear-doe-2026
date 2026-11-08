# runner.ps1 - independent #51 GPU worker for DESKTOP-T1K698V (Windows Task Scheduler, current user, no elevation).
# ASCII only (Windows PowerShell 5.1). Each run:
#   1. single instance (named mutex), cooldown check, time window check
#   2. deterministic precheck.py (no LLM). NO_TASK/LOCKED/BUSY/GH_ERROR -> log and exit, Claude is NOT called
#   3. TASK -> one headless `claude -p` (separate process) that follows WORKER.md for that TASK_ID
#   4. quota / rate-limit text in Claude output -> cooldown (Retry-After / reset time if printed, else 60 min)
# Self-test: if SELFTEST.request exists, run one headless Claude that reads #51 and posts the autonomy test comment.
$ErrorActionPreference = 'Stop'
$Dir     = 'C:\nuclear-doe-operator\ops\worker51'
$Work    = 'C:\nuclear-doe-operator\b-sampler'
$Py      = 'C:\nuclear-doe-operator\desktop-41\.venv-41\Scripts\python.exe'
$Claude  = 'C:\Users\Namee\.local\bin\claude.exe'
$LogF    = Join-Path $Dir 'runner.log'
$RunF    = Join-Path $Dir 'runner-state.json'
$EndAt   = [DateTimeOffset]::Parse('2026-11-08T09:30:00+03:00')   # Stop Coding: no new development after this
function Log($m) { Add-Content -Path $LogF -Value ("{0} pid={1} {2}" -f (Get-Date -Format s), $PID, $m) -Encoding ascii }

$mutex = New-Object System.Threading.Mutex($false, 'Local\nuclear-doe-4070ti-worker')
if (-not $mutex.WaitOne(0)) { Log 'SKIP another runner instance holds the mutex'; exit 0 }
try {
  if (Test-Path $RunF) { $st = Get-Content $RunF -Raw | ConvertFrom-Json } else {
    $st = [pscustomobject]@{ cooldown_until = $null; claude_calls = 0; max_claude_calls = 12; runs = 0 } }
  $st.runs = [int]$st.runs + 1
  function Save { $st | ConvertTo-Json -Depth 5 | Set-Content -Path $RunF -Encoding ascii }

  function Invoke-Claude([string]$prompt, [string]$tag) {
    if ([int]$st.claude_calls -ge [int]$st.max_claude_calls) { Log "SKIP claude call cap reached ($($st.claude_calls))"; return $null }
    $st.claude_calls = [int]$st.claude_calls + 1; Save
    $out = Join-Path $Dir ("claude-{0}-{1}.json" -f $tag, (Get-Date -Format 'yyyyMMdd-HHmmss'))
    $pf = Join-Path $Dir ("prompt-{0}.txt" -f $tag)
    [IO.File]::WriteAllText($pf, $prompt, (New-Object Text.UTF8Encoding($false)))
    $cargs = @('-p', '--output-format', 'json', '--max-turns', '60', '--permission-mode', 'acceptEdits',
              '--add-dir', 'C:\nuclear-doe-operator', '--add-dir', 'C:\ndoe-out',
              '--allowedTools', 'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep')
    Log "CLAUDE start tag=$tag call=$($st.claude_calls) out=$out"
    $p = Start-Process -FilePath $Claude -ArgumentList $cargs -WorkingDirectory $Work -WindowStyle Hidden `
           -RedirectStandardInput $pf -RedirectStandardOutput $out -RedirectStandardError ($out + '.err') -PassThru
    $done = $p.WaitForExit(110 * 60 * 1000)
    if (-not $done) { try { $p.Kill() } catch {}; Log "CLAUDE timeout tag=$tag (killed after 110 min)" }
    $txt = ''
    foreach ($f in @($out, ($out + '.err'))) { if (Test-Path $f) { $txt += (Get-Content $f -Raw -ErrorAction SilentlyContinue) } }
    $code = $p.ExitCode
    Log ("CLAUDE end tag={0} exit={1} bytes={2}" -f $tag, $code, $txt.Length)
    if ($txt -match '(?i)(usage limit|rate.?limit|429|quota|limit reached|resets? at)') {
      $mins = 60
      if ($txt -match '(?i)retry-after["'':\s]+(\d+)') { $mins = [math]::Max(5, [math]::Ceiling([int]$Matches[1] / 60)) }
      $st.cooldown_until = (Get-Date).AddMinutes($mins).ToString('s'); Save
      Log "COOLDOWN quota/rate-limit text detected -> until $($st.cooldown_until)"
    }
    return $txt
  }

  if ($st.cooldown_until -and (Get-Date) -lt [datetime]$st.cooldown_until) { Log "SKIP cooldown until $($st.cooldown_until)"; Save; exit 0 }

  $self = Join-Path $Dir 'SELFTEST.request'
  if (Test-Path $self) {
    Remove-Item $self -Force
    $prompt = @"
AUTONOMY SELF-TEST of the independent Windows Task Scheduler worker (task nuclear-doe-4070ti-worker). You are a NEW
headless Claude Code process started by runner.ps1 (runner PID $PID), not an interactive chat. Do exactly:
1) Run: hostname ; nvidia-smi --query-gpu=name,uuid,memory.used --format=csv,noheader ; git -C C:\nuclear-doe-operator\b-sampler rev-parse --short HEAD
2) Read the last 3 comments of GitHub issue #51 with: "C:\Program Files\GitHub CLI\gh.exe" api "repos/NinjaMoons/nuclear-doe-2026/issues/51/comments?per_page=100" --jq ".[-3:][]|.id"
3) Run: powershell -NoProfile -Command "(Get-ScheduledTask -TaskName nuclear-doe-4070ti-worker | Get-ScheduledTaskInfo).NextRunTime"
4) Write a file C:\nuclear-doe-operator\ops\worker51\selftest-body.md whose first line is <!-- sanya-autonomy-selftest --> and second line:
   SANYA_AUTONOMY_SELFTEST=PASS | HOST=<hostname> | GPU=<name uuid> | SESSION=NEW_HEADLESS_PROCESS (Windows Task Scheduler -> runner.ps1 -> claude -p) | RUNNER_PID=$PID | GIT_SHA=<sha> | GH_READ=PASS (last #51 ids: <ids>) | GH_WRITE=PASS (this comment) | NEXT_RUN=<from step 3>
5) Post it: "C:\Program Files\GitHub CLI\gh.exe" issue comment 51 -R NinjaMoons/nuclear-doe-2026 --body-file C:\nuclear-doe-operator\ops\worker51\selftest-body.md
6) Print the comment URL. Do not do anything else.
"@
    $r = Invoke-Claude $prompt 'selftest'
    if ($r -and $r -match 'issuecomment-(\d+)') { Log "SELFTEST posted comment $($Matches[1])" } else { Log 'SELFTEST no comment URL found in output' }
    Save; exit 0
  }

  if ([DateTimeOffset]::Now -ge $EndAt) { Log 'SKIP after Stop Coding 09:30 MSK'; Save; exit 0 }

  $pre = & $Py -I (Join-Path $Dir 'precheck.py') 2>&1 | Select-Object -Last 1
  Log "PRECHECK $pre"
  try { $j = $pre | ConvertFrom-Json } catch { Log 'PRECHECK unparsable'; Save; exit 0 }
  if ($j.action -ne 'TASK') { Save; exit 0 }

  $t = $j.tasks[0]
  $prompt = @"
You are the independent scheduled #51 GPU worker on DESKTOP-T1K698V (RTX 4070 Ti), GitHub user MVRto01, started by
Windows Task Scheduler (runner.ps1 PID $PID) as a NEW headless Claude Code process. The deterministic precheck found
TASK_ID=$($t.task_id) (comment $($t.url), author $($t.author)); current team phase = $($j.phase); tick.lock is held for you.
Follow C:\nuclear-doe-operator\ops\worker51\WORKER.md exactly for this one TASK_ID: read the card and the newest
#51/#49/#8 comments (a newer NO_GO or withdrawal wins), check TO is exactly @MVRto01/DESKTOP-T1K698V, science GO,
data rights (native TRAIN512 / no TEST decoding: use the no-TEST wrappers in C:\nuclear-doe-operator\r19\tphase_head
or research/wc_stem_pilot as appropriate), the phase gate (new training only 02:00-04:30 MSK, finish by 05:15, nothing
after the 06:00 freeze), and that no CLAIM / output dir / GPU python PID already exists for it. Then post CLAIMED,
run at most ONE GPU job in its own directory (wait for it to finish within this session), save SHA + logs, upload a
private asset if the card asks, post DONE/NO_GO/BLOCKED, update C:\nuclear-doe-operator\ops\worker51\state.json
processed[$($t.task_id)] and delete C:\nuclear-doe-operator\ops\worker51\tick.lock. Never decode TEST, never
overwrite V1/V2/4ch, no per-input Adam/GS at inference, never lower acceptance criteria, never submit the contest form,
never change repo visibility or access, no paid actions.
"@
  $r = Invoke-Claude $prompt ('task-' + ($t.task_id -replace '[^A-Za-z0-9_.-]', '_'))
  if (Test-Path (Join-Path $Dir 'tick.lock')) {
    $age = ((Get-Date) - (Get-Item (Join-Path $Dir 'tick.lock')).LastWriteTime).TotalMinutes
    Log ("AFTER task: tick.lock still present (age {0:N0} min); precheck treats it as stale after 50 min" -f $age)
  }
  Save
} finally {
  $mutex.ReleaseMutex() | Out-Null
}
