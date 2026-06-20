# Contributing to PyQCA

Thank you for considering a contribution to PyQCA.

## Development Setup

```bash
git clone https://github.com/t-yamsaki/PyQCA.git
cd PyQCA
python -m pip install -e ".[dev]"
```

Run the quality checks before submitting a pull request:

```bash
python -m ruff check .
python -m ruff format . --check
python -m isort . --check-only
python -m flake8 .
python -m pytest tests
```

## Contribution Scope

- Keep changes focused and include tests for behavior changes.
- Use the existing public APIs and module boundaries where practical.
- Document new user-facing behavior in both `README.md` and `README.ja.md`
  when applicable.
- Do not commit generated caches, build artifacts, datasets containing
  confidential information, or vendored dependencies without prior agreement.

## Licensing

PyQCA is licensed under the MIT License.

By submitting a contribution, you certify that:

1. you have the right to submit the contribution;
2. the contribution is your original work or is used under a compatible
   license with all required notices included;
3. you agree that your contribution is provided under the project's MIT
   License;
4. you have disclosed any code derived from another project, including its
   source URL, version or commit, copyright holder, and license.

Do not copy or translate GPL-licensed source code into PyQCA. Descriptions of
algorithms, published formulas, and independently implemented methods are
welcome, but the implementation must be original or compatible with the MIT
License.

If a contribution adapts MIT-, BSD-, or Apache-licensed code, preserve all
required attribution and update `THIRD_PARTY_NOTICES.md` and
`PROVENANCE_AUDIT.md`.

## Research Attribution

Methodological contributions should cite the relevant books, articles, and
software. Where a feature reproduces or adapts another package's workflow,
document both the conceptual relationship and the source-code relationship.

## Pull Requests

Include:

- a concise description of the change;
- tests and their results;
- documentation updates;
- provenance and licensing notes when external material informed the change.

All contributions are reviewed for correctness, API consistency,
reproducibility, and license compatibility.
