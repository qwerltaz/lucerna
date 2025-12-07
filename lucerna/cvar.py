"""Configuration variables."""

from pathlib import Path

data_dir: Path = Path("../data")
resources_dir: Path = Path("../resources")
licma_dir: Path = Path(__file__).parent.parent / "licma"
