import json
from pathlib import Path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config_path = path.resolve()
    root_dir = config_path.parent.parent

    config["root_dir"] = str(root_dir)
    inventory_path = Path(config["inventory_csv"])
    output_path = Path(config["output_dir"])
    config["inventory_csv"] = str((root_dir / inventory_path).resolve())
    config["output_dir"] = str((root_dir / output_path).resolve())
    return config
