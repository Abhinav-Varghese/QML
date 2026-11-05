import numpy as np
import os
from datetime import datetime
import yaml
import uuid
from omegaconf import OmegaConf
import torch
from sklearn.metrics import roc_curve

# ------ Directory Creator ------ #
def createRunDir(cfg):
    """
    Creates a new run folder with the current date and time, as well as a random UUID and custom tag (if passed).
    Also attaches a scan-id if performing a parameter scan, which is put in a subfolder
    """
    tag = f":{cfg.misc.folder_tag}" if cfg.misc.get("folder_tag") else ""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    if cfg.misc.get("scan_id"):
        scan_dir   = f"scan:({cfg.misc.scan_id})"
        run_name   = f"{cfg.misc.scan_index}{tag}"
        run_folder = os.path.join(cfg.data.save_path, date_str, scan_dir, run_name)

        # Also pass/log scanned params
        scan_keys   = cfg.misc.scan_keys.split(",")
        scan_params = {k.split(".")[-1]: OmegaConf.select(cfg, k) for k in scan_keys}

        OmegaConf.set_struct(cfg, False)
        OmegaConf.update(cfg, "misc.scan_params", scan_params)
        OmegaConf.set_struct(cfg, True)

        OmegaConf.update(cfg, "misc.scan_keys", "")
    else:
        uid        = str(uuid.uuid4().hex)[:5]
        run_name   = f"{now.strftime('%H-%M-%S')}:({uid}){tag}"
        run_folder = os.path.join(cfg.data.save_path, date_str, run_name)

    os.makedirs(run_folder, exist_ok=True)

    ## Save config used (subRNGs are not included)
    cfg_to_dump = OmegaConf.to_container(cfg, resolve=True)

    with open(f"{run_folder}/configUsed.yml", "w") as f:
        yaml.safe_dump(cfg_to_dump, f, sort_keys=False)
    
    # Print on completion
    print(f"- Run Directory Created: {run_folder}")
    return run_folder

class saveUsefulInfo():
    """
    Class for easily saving useful/quick info to a summary.yml file
    """
    def __init__(self,run_path):
        self.path=os.path.join(run_path,'summary.yml')
        self.data={}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data=yaml.safe_load(f) or {}
                
    def add(self,key,value):
        self.data[key]=value
        with open(self.path,'w') as f:
            yaml.safe_dump(self.data,f)

def finishRunDir(run_path):
    """
    Updates run path with "done" upon finishing all training/plotting.
    """
    new_path = run_path + ":Done"
    os.rename(run_path, new_path)
    return

def createSubRNGS(cfg):
    """
    This function creates sub random number generators, so each "logical" section has its own randomness. Otherwise, the sections become dependent,
    e.g., if process A and B use randomness, then changes to A affect the randomness used in B.
    Explicitly, a larger circuit (with more randomised parameters) could affect data shuffling.
    """

    # Create master rng from seed
    master_rng = np.random.default_rng(cfg.misc.random_seed)

    # Use this master rng to create other seeds
    seeds = {
        "data":    int(master_rng.integers(2**32)),
        "circuit": int(master_rng.integers(2**32)),
        "training":   int(master_rng.integers(2**32))}

    # These seeds then create subRNGs
    rngs = {
        "data": np.random.default_rng(seeds["data"]),
        "circuit": torch.Generator().manual_seed(seeds["circuit"]),
        "training": torch.Generator().manual_seed(seeds["training"]),
    }
    return rngs

def YoudenJ(losses,labels):
    
    fpr, tpr, thresholds = roc_curve(labels, losses)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold = thresholds[best_idx]

    return best_threshold

def decode_arg(arg):
    """
    Decode +-delimited list values back to OmegaConf format.
    e.g. 'circuit.trash_modes=3+4' → 'circuit.trash_modes=[3,4]'
    Scalars and non-list args pass through unchanged.
    Also, trash mode lists with single int [3] etc are handled, despite having no +.
    """
    if "=" not in arg:
        return arg

    key, value = arg.split("=", 1)

    if value.startswith("L:"):
        value = value[2:]
        if value == "":
            value = "[]"
        else:
            value = "[" + ",".join(value.split("+")) + "]"

    return f"{key}={value}"