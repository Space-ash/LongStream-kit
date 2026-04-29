import copy
import os
from typing import Dict, Optional

import yaml


def default_preprocess_config_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs",
        "preprocess.yaml",
    )


def load_preprocess_config(config_path: Optional[str] = None) -> Dict:
    path = config_path or default_preprocess_config_path()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: Dict, overrides: Dict) -> Dict:
    merged = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
