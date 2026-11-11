# SIPPO

Research code for learning executable retrieval workflows from sibling execution
outcomes, built on [verl](https://github.com/volcengine/verl). The repository
contains the workflow runtime, terminal-outcome RL, Planner Sibling Contrast
(PSC), experiment configuration and tests.

## Status

This is a source snapshot of ongoing research, not a certified reproduction
package. The experimental NR-PSC implementation is under development and has
not completed end-to-end validation. Do not interpret its presence as a
reported training result.

## Getting Started

Python dependencies are listed in `requirements.txt` and `pyproject.toml`.
Install the package in a suitable Python and accelerator environment with:

```bash
pip install -e .
```

Dataset files, retriever indexes, model checkpoints, credentials, and private
cluster configuration are not included. Training and evaluation require those
dependencies to be configured for the local environment.

The prior project README is preserved as `README_JADE_ORIGINAL.md`; some legacy
paths and commands in that document may not be applicable to this snapshot.
See `LICENSE` and `Notice.txt` for upstream notices.
