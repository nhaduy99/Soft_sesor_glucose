# AGENTS.md

## Project overview
This project is about: organizing Emilie soft-sensor raw data, building matched EEM/Raman/HPLC inventories, generating scientific visualizations, exporting ML-ready spectroscopy features, and training interpretable soft-sensor workflows for monosaccharide prediction.

Current modelling targets are `rhamnose_gL`, `xylose_gL`, and `glucose_gL`. Supervised results currently use standards and known spikes parsed from treatment labels. Culture-sample prediction still requires quantitative HPLC monosaccharide targets.

## Start-of-session routine
Before coding, always:
1. Read `PROGRESS.md`.
2. Read `PROJECT_CONTEXT.md`.
3. Read `DECISIONS.md`.
4. Read `TODO.md`.
5. Read `SESSION_HANDOFF.md`.
6. Check `git status`.
7. Inspect recent commits with `git log --oneline -5`.
8. Summarize the current state before making changes.

## Development commands
- Install dependencies: `pip install -r rhamnose_ml/requirements.txt`
- Rebuild inventory: `python build_enriched_inventory.py`
- Export features: `python export_rhamnose_features.py`
- Run unsupervised exploration: `python explore_features_unsupervised.py`
- Run supervised monosaccharide search: `python train_monosaccharide_softsensor.py`
- Run scaffold baseline: `python rhamnose_ml/scripts/train_baseline.py --config rhamnose_ml/config/defaults.json`
- Run tests: `python -m compileall rhamnose_ml/src rhamnose_ml/scripts`
- Run syntax checks: `python -m py_compile build_enriched_inventory.py visualize_eem_raman.py export_rhamnose_features.py explore_features_unsupervised.py train_monosaccharide_softsensor.py`

## Coding rules
- Keep changes minimal and focused.
- Do not rewrite working modules unless necessary.
- Add comments only where they clarify non-obvious logic.
- After changes, run the relevant tests.

## Completion rules
Before finishing a task:
1. Summarize changed files.
2. Report tests run and results.
3. Update `PROGRESS.md`.
4. Update `TODO.md` if tasks changed.
5. Update `SESSION_HANDOFF.md` with the latest status, commands run, and next action.
6. Commit and push when the user asked for repo-visible progress or project state changes.

At the end of every task, update `SESSION_HANDOFF.md` so the next Codex session can continue without chat history.
