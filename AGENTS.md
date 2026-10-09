# xarray-fits

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), recorded on each ticket's `Status:` line. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `CONTEXT.md` plus `docs/adr/`. See `docs/agents/domain.md`.

## Tooling

- Install: `uv sync --all-groups`
- Test: `uv run pytest -s -vvv tests/ -Werror`
- Lint, format and type-check: `uv run pre-commit run -a`
- Code style: 2-space indentation, line length 88, Google docstrings
- Changelog: `docs/source/changelog.rst`, newest first, entries end with `(:pr:`NNN`)`
- Release: `uv run tbump X.Y.Z` bumps `pyproject.toml`, `docs/source/conf.py` and `xarrayfits/__init__.py`
