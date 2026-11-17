pt *args="":
  uv run ptpython {{args}}

slides *args="":
    FIGURE_PROFILE=slides uv run pytask {{args}}

journal *args="":
    FIGURE_PROFILE=journal uv run pytask {{args}}

