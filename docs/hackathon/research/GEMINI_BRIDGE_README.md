# Claude Code ↔ Gemini research bridge · Grigoriy · 07–08 Nov 2026

**Status: CODE READY IN GITHUB; NOT CONNECTED TO GRIGORIY'S MAC YET.** A GitHub file does not install CLI, log in to Google, or create scheduled polling.

## Why this exists

Pavel reports ONE usable Gemini web account, with only ~three useful questions left; second web account has reached limit. Gemini Apps screenshot: 67% 5h **USED**, 17% weekly USED, reset **03:53 local UI time** (MSK only if user's system zone is MSK). The new primary engine is **Claude Opus independent science**, not access to Gemini. Use the remaining browser queries carefully; after expiry Opus continues itself. The Gemini web UI has no native autonomous machine-to-machine feed to a Claude Code terminal.

A practical **programmatic** feed is possible through a SEPARATE Gemini CLI product: Claude Code can run a subprocess, which sends a short research prompt to Gemini CLI headless and obtains its JSON reply. In `docs/hackathon/research/gemini_bridge.py` we implement this one-shot handoff. It is **not a browser bridge** and is **not guaranteed to use the same Gemini Apps quota or model**. Gemini CLI / Code Assist authentication, supported models, quotas and privacy terms may differ; verify locally. If the user wants **exactly the three remaining Gemini web UI calls**, use the browser and paste original raw outputs once, without pretending they are automatable via the CLI.

Official reference links:
- Gemini CLI install: https://geminicli.com/docs/get-started/installation/
- Gemini CLI authentication: https://geminicli.com/docs/get-started/authentication/
- Headless `--prompt --output-format json`: https://geminicli.com/docs/cli/headless/
- Claude Code as a shell/MCP user: https://support.claude.com/en/articles/14554922-claude-code-user-faq
- Gemini Apps limits are separate: https://support.google.com/gemini/answer/16275805?hl=en

## Step 1 — quick nonintrusive probe on Grigoriy's macOS

From the existing checked-out repo (avoid altering running research worktree):
```bash
git fetch origin main
git show origin/main:docs/hackathon/research/GEMINI_BRIDGE_README.md
node --version
gemini --version
```

If `gemini` is absent, ask whether a ONE-TIME install is worth the deadline risk; do not claim bridge connected until actually verified:
```bash
npm install -g @google/gemini-cli
gemini
```
Only Grigoriy should finish login in his browser, using his OWN authorized Google account. Never send Google credentials, private API keys or session cookies to Claude/ChatGPT/GitHub. Personal Google AI Pro generally supports sign-in but CLI quotas differ from the web UI. API key or Vertex AI is an alternative route with potentially separate costs and authorizations, NOT automatically billed under the web-app subscription.

## Step 2 — verify a tiny real headless response, using a non-sensitive prompt

```bash
gemini --prompt "Reply with only GEMINI_BRIDGE_OK" --output-format json
```

Check that a JSON object contains `response` and usage stats. Check actual model/plan/quota on the CLI; do not assume the highest web model is used. If command fails with subscription/429/auth/tool error, **stop** and fall back to Claude Opus. Do not install unofficial MCP plugins or repeatedly retry against the 5h cap.

## Step 3 — prompt → Gemini → saved output → Claude, in ONE agent turn

Create a brief **non-secret** physics research prompt, no raw private dataset/TEST or credentials:
```bash
mkdir -p .local-gemini-inbox
printf '%s\n' 'DOE problem: four amplitude targets at 24.1/25.3/26.5/27.7mm. Propose ONE falsifiable analysis of B plane ghosting that does not duplicate our completed experiments. Do not invent metrics.' > .local-gemini-inbox/one_question.md
python docs/hackathon/research/gemini_bridge.py --prompt .local-gemini-inbox/one_question.md --out .local-gemini-inbox/answers
```

**Run this command INSIDE Grigoriy's Claude Code through its normal shell tool**, not in ChatGPT. The bridge writes `prompt.md`, `raw_stdout.json`, `reply.md`, `receipt.json` and local diagnostics in a unique timestamped directory and returns:
- `OK` → Claude reads `reply.md`, cross-checks all claims, writes scientific `IDEA_CARD`, posts #49.
- `GEMINI_NOT_INSTALLED`/ `QUOTA_OR_RATE_LIMIT` / `CLI_ERROR` / `PARSE_ERROR` / `TIMEOUT` → no more Gemini calls; Claude continues autonomous work, logs cause and next task. **No retry loop**. The bridge intentionally runs Gemini with a temporary empty working directory, not the private repo.

**IMPORTANT**: add `.local-gemini-inbox/` to the user's local git ignore (`.git/info/exclude`) or keep output outside the repository; do NOT commit local stderr/config. Only after human/agent checks privacy may safe sanitized prompt and reply be archived into `docs/hackathon/research/gemini/` on Grigoriy's branch. Keep full original answer and exact source.

## Monitoring and failure isolation

- While Claude Code is running, it may itself decide whether a new Gemini query is useful and call the bridge. The Gemini CLI is invoked only on request; **there is no separate Gemini watcher** and no two live bots chatting continuously. That is deliberate for quotas and safety.
- Claude CLI quota exhaustion is independent. Use the approved Claude Desktop scheduled task or deliberate manual resume after its reset, reading `NEXT_STEP.md`; if the CI/OS tasks are running, Python/GPU training can continue without any LLM call.
- If Gemini web resets at ~03:53 MSK, the 04:00 research cutoff is already near: use newly available requests for independent scientific QC or presentation/defense, NOT for starting a new long training job. Freeze package/defense evidence 05:30 MSK.
- Any machine-to-machine interaction with API services still consumes the relevant provider quotas; do not promise uninterrupted operation across exhausted quotas.

**No install/run claimed.** Only Grigoriy on his actual Mac can test and record `GEMINI_BRIDGE_TEST=PASS/FAIL; MODEL; QUOTA/RESET/TZ; RUNTIME` in [#49](https://github.com/NinjaMoons/nuclear-doe-2026/issues/49).
