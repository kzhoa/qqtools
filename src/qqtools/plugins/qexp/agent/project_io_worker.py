"""Stable subprocess entry point used by agents across package upgrades."""

from .project_io.worker import main

if __name__ == "__main__":
    raise SystemExit(main())
