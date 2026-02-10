import json
from pathlib import Path


def write_attribute_stats(dataset_root: Path, stats: dict):
    stats_dir = dataset_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    out_path = stats_dir / "attributes.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
