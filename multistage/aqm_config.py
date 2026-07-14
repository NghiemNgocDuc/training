# aqm_config.py
# Configuration for the two-stage learning approach using AQM dataset.

# -------------------------------------------------------------------------
# INPUT FEATURES
# -------------------------------------------------------------------------
# Atom Type - tells the model what element each atom is. 
# Used in both Stage 1 (vacuum) and Stage 2 (solvated).
ATOM_TYPE_FEATURE = "atNUM" 

# Atom Coordinates - the 3D shape of the molecule.
# We use the water-relaxed shape for both models to ensure consistency.
ATOM_COORDS_FEATURE = "atXYZ" 

# -------------------------------------------------------------------------
# STAGE 1: VACUUM MODEL TARGETS
# -------------------------------------------------------------------------
# Dataset to use for Stage 1: AQM-gas.hdf5
# Target for Stage 1 training
VACUUM_ENERGY_TARGET = "ePBE0+MBD"

# Target for Stage 1 training, alongside vacuum energy
VACUUM_FORCES_TARGET = "totFOR"

# -------------------------------------------------------------------------
# STAGE 2: CORRECTION MODEL TARGETS
# -------------------------------------------------------------------------
# Dataset to use for Stage 2: AQM-sol.hdf5
# Target for Stage 2 training (true solvated energy)
SOLVATED_ENERGY_TARGET = "ePBE0+MBD"

# Target for Stage 2 training, alongside solvated energy
SOLVATED_FORCES_TARGET = "totFOR"

# Optional extra target for Stage 2
SOLVATION_FREE_ENERGY_TARGET = "eSOLV"

# -------------------------------------------------------------------------
# SUMMARY OF WORKFLOW:
# 1. Train vacuum model on (AQM-gas ATOM_TYPE, ATOM_COORDS -> VACUUM_ENERGY, VACUUM_FORCES)
# 2. Freeze vacuum model.
# 3. For each molecule, feed AQM-sol ATOM_TYPE and ATOM_COORDS into frozen vacuum model -> get baseline guess.
# 4. Train correction model to predict (SOLVATED_ENERGY - baseline guess) and forces.
# -------------------------------------------------------------------------
