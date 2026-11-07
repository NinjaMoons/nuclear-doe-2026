# NEXT_STEP — @ggmaster913 (Mac M1, scientific gateway #49) — updated 2026-11-07 23:43 MSK

- Running jobs: R18 pilot on this Mac (background; outputs ~/Desktop/r12-runs/r18_s0_{control,bpmix}, logs *.log).
- Branch: research/49-gemini-intake (this file), worktree ~/Desktop/g49. Audit dirs: ~/Desktop/stem-audit (clean py3.12 venv venv312, audit script audit/stem_audit_ckpt.py), ~/Desktop/sanya-audit, ~/Desktop/qc47.
- Pending: (1) raw Gemini G1/G2 answers from the user -> commit verbatim here, verify, IDEA_CARDs; (2) review Sanya #43 distillation loss/labels (DISTILL_NEUTRAL) and LR-recipe result when DONE; (3) independent cross-host acceptance of any new candidate (stem_audit_ckpt.py + VAL16 replay + #47 sheets).
- Time gates MSK: no new training starts 02:00; research stop 04:00; freeze 05:30; submission 07:30-08:00.
- Recovery: session-only 15-min GitHub check in this open Claude Code Desktop session (dies with the session); otherwise RESUME_MANUALLY: open Claude Code in ~/Desktop, read this file and #49.

## 02:25 MSK 08.11
- R18 final: NO_GO (seed-mean dP +.0023); posted #49/#43/#8. RESULT_R18.md @ efd2c15.
- G4 verified (G4_verification.md, M0–M4 test).
- R19 (learned target-plane phase) pre-registered, branch research/r19-target-phase @ a07b6e9. READY card in #51 (machine header 6714241999) for Sanya's 4070 Ti.
- Mac: R19 tphase s0 on CPU → ~/Desktop/r12-runs/r19_s0_tphase_mac. Compare against r18_s0_control (the same code/seed/sampler) with analyze_r12 + seedmean_r18.
- R20 (contrastive loss) belongs to the main session; I do the independent acceptance of R20, it does R19's.
- Gates: new training starts ≤ 04:30 (Sanya ≤ 04:15), finish by 05:15, acceptance 05:15–06:00, freeze 06:00.
