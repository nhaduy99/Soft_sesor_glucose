# Literature Input Folder

Use this folder for rhamnose method-review inputs.

## Local PDFs
Place PDF papers in:

`literature/papers/`

The review tool will extract text from these PDFs with PyMuPDF when available.

## Seed references
Add known papers, DOIs, URLs, or notes to:

`literature/seed_references.csv`

The seed file can be used even when web discovery is disabled.

## Run
```powershell
conda run -n base python literature_methods_review.py
```

Enable optional Crossref discovery with:

```powershell
conda run -n base python literature_methods_review.py --web
```
