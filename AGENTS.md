# AGENTS.md

## Project overview
This project is about: organizing Emilie soft-sensor raw data, building matched EEM/Raman/HPLC inventories, generating scientific visualizations, exporting ML-ready spectroscopy features, and preparing baseline Rhamnose prediction workflows.

## Start-of-session routine
Before coding, always:
1. Read `PROGRESS.md`.
2. Read `TODO.md`.
3. Check `git status`.
4. Inspect recent commits with `git log --oneline -5`.
5. Summarize the current state before making changes.

## Development commands
- Install dependencies: `pip install -r rhamnose_ml/requirements.txt`
- Run app: `python rhamnose_ml/scripts/train_baseline.py --config rhamnose_ml/config/defaults.json`
- Run tests: `python -m compileall rhamnose_ml/src rhamnose_ml/scripts`
- Run lint: `python -m py_compile build_enriched_inventory.py visualize_eem_raman.py export_rhamnose_features.py explore_features_unsupervised.py`

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
