"""
backend/module4 — Module 4 (NutriIngredientNet): food photo -> nutrition.

Kept as its own package so Module 4's code stays separate from the Module 1 +
Module 2 workout pipeline it shares a server with. It is NOT part of that
pipeline: no shared state, no shared frames, its own page and its own routes.

Unlike Module 1 and Module 2, Module 4 imports nothing first-party (no bare
`import config`, no `from src import ...`), so it has no import collision with
either of them and can live in this process directly — no worker subprocess
needed.
"""
