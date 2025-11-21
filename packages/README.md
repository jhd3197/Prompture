# Wrapper packages

Each lightweight wrapper that depends on Prompture lives inside its own
subdirectory under `packages/`. Currently shipping wrappers:

- `llm_to_json` – returns parsed JSON objects.
- `llm_to_toon` – emits ultra-compact TOON strings.

A wrapper directory normally contains:

- `pyproject.toml` – metadata for the package on PyPI
- `README.md` – package specific documentation
- the actual Python package folder (for example `llm_to_json/`)
- optional helper scripts/tests that only belong to that wrapper

Example layout:

```
packages/
  llm_to_json/
    pyproject.toml
    README.md
    llm_to_json/__init__.py
  llm_to_toon/
    pyproject.toml
    README.md
    llm_to_toon/__init__.py
```

When you add another wrapper:

1. Create a new directory following the example above.
2. No extra configuration is needed for `.github/scripts/update_wrapper_version.py`
   because the script automatically scans every subdirectory that contains a
   `pyproject.toml`.
3. Add build/publish steps for the new package in
   `.github/workflows/publish.yml` (or extend the existing bash loop).
