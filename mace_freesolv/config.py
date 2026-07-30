import os

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVATION_GNN = os.path.join(ROOT, "aqm-spice2")
HDF5_PATH = os.path.join(ROOT, "freesolv_conformers.hdf5")
MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "mace")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# MACE model
MACE_MODEL_SIZE = "medium"  # small | medium | large
MACE_R_MAX = 5.0
MACE_MAX_NEIGHBORS = 32

# Fine-tuning
EPOCHS = 500
LR = 1e-4
LR_MIN = 1e-7
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
PATIENCE = 50
VAL_SPLIT = 0.2
SEED = 42
N_FOLDS = 5
WARMUP_EPOCHS = 10
LOSS_TYPE = "mse"  # mse | huber

# Targets (FreeSolv experimental dG in kcal/mol)
EV_TO_KCAL = 23.0605

# LoRA
USE_LORA = False
LORA_RANK = 32
LORA_ALPHA = 2.0
LORA_UNFREEZE_READOUTS = True
LORA_UNFREEZE_SKIP_TP = True

# Architecture
FREEZE_ATOMIC_ENERGIES = True
FREEZE_INTERACTIONS = False
