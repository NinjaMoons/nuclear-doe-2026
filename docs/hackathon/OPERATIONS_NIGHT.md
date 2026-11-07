# Autonomous night command board · 07→08 Nov 2026 (MSK)

**Canonical evidence:** [defense index](defense/DEFENSE_INDEX.md) · [Gemini↔Claude research handoff](research/GEMINI_TO_CLAUDE.md). Both created on isolated ops branch. Avoid changing production/main without an explicit release gate. Ensure scientific and defense docs are cherry-picked/merged or referenced by branch prior to preparing slides.

**Ownership is not a prison.** Owner = accountability and context. After DONE, each Claude chooses the next non-overlapping highest-value task according to current time, quota, resources and submission needs. Report TRUE host/GPU status; absence of a comment alone is not proof of idle execution.

| Owner / host | Current ACTIVE (do not interrupt) | NEXT while that runs / upon DONE | Secondary lane |
|---|---|---|---|
| **Sanya @MVRto01** · RTX4070Ti desktop | #43 3 width/depth variants × 2 seeds, 12000 updates; report actual finish/metrics/checkpoints | If capacity winner: hand to Matvey for candidate packaging and Grigoriy for cross-host audit; otherwise pilot train-only teacher-distillation iff p90 feasible, licensing safe | #41 provenance and reproducible reports, visual B-specific inspection |
| **Sanya laptop** · hardware unverified | **Do not assume CUDA** or start second process on same machine without physical host check | If true independent GPU + VRAM available: assigned non-duplicative short experiment *after host ACK and no shared dependencies*; else CPU data audit/metric exports | Assists defense, dry run |
| **Matvey @playerr456** · reportedly capable laptop, prior session only CPU | V2 package asset721714285 complete independent PASS; possible read-only QA | Measure actual CUDA/MPS/CPU+VRAM, jobs, quota/reset; while Sanya trains, prepare **train-only teacher/student pilot design**, anti-wrap losses, teacher storage/spec, synthetic tests, GPU budget; no unapproved simultaneous old-run replay | If CUDA sufficiently fast, run matched independent experiment on distinct host/worktree (not Sanya's capacity or same GPU) only with prereg/GO gates. If CPU only, handle packaging, CI, visuals, 1-command repro, slides |
| **Grigoriy @ggmaster913** · Mac M1 CPU | Physics / independent checks; two Gemini research streams | Ingest **full raw Gemini outputs**, critique with actual code, create reproducible GitHub IDEA_CARD, dispatch experiments to free machines; independent visual/physics audits of new candidate | Defense scientific explanation and physics diagram; no repeated failed D4/R15 |
| **Coordinator @NinjaMoons** · available host unknown | P0 #8 submission and #47 defense, version choice, official organizer clarification, real upload receipt | Own evidence index, git branch/PR, slide outline/references, collect asset SHA, verify deck tracks selected checkpoint, nominate presenter and submitter | Route jobs and resolve conflicts, protect existing v1/v2/4ch packages |
| **Pavel** | Prioritize actual organizer contact / final team call | Escalations for disallowed data, unresolved rules, wrong team identity; no micromanaging safe research | Decide on public explanation and presentation style |

## One source of truth and collaboration signals

- **Git** (small source/report files): docs, training scripts, config, `PLAN.md` prereg before jobs, `RESULT.md`, replay command, slide source, prompts, Gemini raw replies, manifests, plot scripts, metrics CSV/JSON, speaker notes and evidence index.
- **Private GitHub Releases** (large blobs): checkpoints, archives, full raw DOE/recon PNG bundles, contact-sheet ZIP. Always track exact asset ID, name, byte size, full SHA256 and commit/manifest ref in Git docs; private release is part of GitHub, not git commit history.
- **GitHub Issues** (#43 train, #40 pack, #47 visuals, #8 submit and Gemini intake issue) are task/event logs. Link to immutable artifacts. A GitHub issue comment isn't a persistent file in a Git checkout; write the actual `.md` to repo/branch.
- **NO overwrites/force-push** of protected v1/v2/4ch release assets, checkpoints or test sets. One host/process one primary ACTIVE. Multiple GPU hosts can run independent experiments after actual resource audit.
- Hand-off schema:
  `STATUS | OWNER | HOST/JOBS | BRANCH/COMMIT | THEORY/ONE_FACTOR | TRAIN_SPLIT | MODEL_SHA | VAL_P/SSIM_ALL_A_B_WORST | B_OWN_4x4/GHOSTING | 4_PLANE_RAW_SHEET + CHECKSUM | MATCHED_CONTROL | WALL_P50/P90 | AI_QUOTA/RESET_OR_UNKNOWN | NEXT_ACTIVE | BLOCKER`.
- Proposals are not empirical results, estimated ETA isn't a measured job finish, a teacher upper bound is not neural inference, and per-input Adam during competition inference is forbidden. WhiteCollars is not approved for inclusion in final.
- **Parallel idea selection**: Grigoriy triages; Sanya owns current capacity job, Matvey may design distillation/test contract while the GPU runs. If both propose same pilot, assign ONE runtime owner and one independent reviewer. Keep exploration short and choose based on B visual isolation, not only VAL P.

## Time/risk control

- 23:00 MSK call: read actual status and assign free GPUs/hosts. Confirm delivery channel and true organization receipt requirements.
- After 02:00: no new research training job starts; implement/review/defense only.
- 04:00: no active research job; freeze candidate choice for independent review.
- 05:30: package/deck source and evidence freeze; late candidate must not silently replace frozen package or slide figures.
- Prefer official submission with real confirmation by 07:30–08:00, keep buffer for 09:30 team target, absolute organizer 10:00. Submission decision must use official rules and authorized owner.
- Report real visible Claude/Gemini quotas and resets, else UNKNOWN; don't conserve calls when resources available but keep one complete independent P0 acceptance/release pass in reserve.

## P0 defense acceptance before slide creation

The deck can be authored by **Claude Code OR Gemini**, depending on available quotas. It must be assembled from `defense/DEFENSE_INDEX.md` + actual linked artifacts, not from an LLM's ungrounded memory. The deck's source, presenter notes, editable presentation and exact screenshot/figure SHA should be in git/private GitHub release and in #47. Show 4 true amplitude reconstructions and explain B ghosting. A scientifically honest failure example is a strength, not a flaw to conceal.
