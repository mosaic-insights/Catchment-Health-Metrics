# Contributing

Thanks for considering a contribution!

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .[dev]
pre-commit install
```

## Tests

```bash
pytest -q
```

## Coding style

- Follow PEP8 and run `ruff --fix .` locally.
- Keep functions small and well-commented.
- Prefer pure functions; avoid side effects when possible.
