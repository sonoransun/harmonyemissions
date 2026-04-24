"""Session-wide pytest configuration.

- Forces the ``Agg`` matplotlib backend so ``plot_*`` helpers don't try to
  open a window on CI.
- Re-registers the custom markers defensively for environments where the
  ``pyproject.toml`` section isn't honoured.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")


def pytest_configure(config):  # noqa: D401 - pytest hook
    for marker in ("slow", "gpu", "mpi", "benchmark"):
        config.addinivalue_line("markers", f"{marker}: see pyproject.toml markers section")
