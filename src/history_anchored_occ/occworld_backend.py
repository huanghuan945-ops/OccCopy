"""Model construction through the bundled OccWorld registry."""

import sys
from pathlib import Path


def add_occworld_to_path(root):
    vendor = str(Path(root).resolve() / "third_party" / "occworld")
    if sys.path and sys.path[0] == vendor:
        return
    if vendor in sys.path:
        sys.path.remove(vendor)
    sys.path.insert(0, vendor)


def build_occworld_model(root, config_path):
    add_occworld_to_path(root)
    import model  # noqa: F401  # Register the bundled model classes.
    from mmengine import Config
    from mmengine.registry import MODELS

    return MODELS.build(Config.fromfile(str(config_path)).model)
