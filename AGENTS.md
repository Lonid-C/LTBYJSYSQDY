# Project working agreement

## Colab-first execution

- Future mathematical-modeling study and experiment runs for this project must be executed in Google Colab through the local `colab` CLI unless the user explicitly asks otherwise.
- Treat the local workspace as the source of truth. Upload the current project inputs to a fresh named Colab session before a run.
- After every Colab notebook run, download the executed `.ipynb` and keep the complete cell outputs in the local `code/` notebook. Validate that every code cell executed and that no error output exists.
- Sync generated result tables, figures, and submission workbooks back to the local workspace when the Colab run changes them.
- Use only historical information available at each simulated decision time; do not introduce data leakage during remote execution.
- Do not store OAuth tokens, Colab session state, or other credentials in this repository.
- Stop the Colab VM after outputs have been synchronized and verified; do not leave idle sessions consuming compute units.

