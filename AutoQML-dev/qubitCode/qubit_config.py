from pathlib import Path
from omegaconf import OmegaConf
import sys

_root = Path(__file__).parent.parent

_config_path = Path(__file__).parent / "qubit_config.yaml"  # default
for arg in sys.argv[1:]:
    if arg.startswith("--config="):
        _config_path = Path(arg.split("=", 1)[1])
        break

cfg = OmegaConf.load(_config_path)
cfg.data.save_path = str(_root / "outputs")

OmegaConf.set_struct(cfg, True) # This throws an error if typo ".circoit" etc, won't create new entry!

CONFIG = cfg