"""MACE-OFF23 "medium" architecture config — EXTRACTED FROM THE LOADED MODEL.

Every value below was read directly off the object returned by
    mace_off("medium", return_raw_model=True, device="cpu")
(mace 0.3.16) on 2026-08-01, NOT taken from any paper or docs. The
introspection script is scratch/introspect_mace_off.py and the full
machine-readable dump is scratch/mace_off23_medium_introspection.json.

Checkpoint file: ~/.cache/mace/MACE-OFF23_medium.model
Top-level class: mace.modules.models.ScaleShiftMACE (1,428,368 params)

NOTE: the loaded checkpoint is a ScaleShiftMACE (MACE + scale/shift
wrapper). For from-scratch construction we rebuild the inner MACE with
these exact args and wrap it the same way (scale=1.0, shift=0.0 for a
fresh model).
"""

# --- extracted from model.r_max buffer ---
R_MAX = 5.0  # buffer on MACE module

# --- extracted from model.num_interactions buffer ---
NUM_INTERACTIONS = 2  # buffer on MACE module

# --- extracted from model.atomic_numbers buffer ---
# [1, 6, 7, 8, 9, 15, 16, 17, 35, 53] == data.py ELEMENT_ATOMIC_NUMBERS (exact match)
ATOMIC_NUMBERS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
NUM_ELEMENTS = len(ATOMIC_NUMBERS)  # 10

# --- radial embedding: extracted from model.radial_embedding ---
NUM_RADIAL_BASIS = 8  # radial_embedding.out_dim == len(bessel_fn.bessel_weights); BesselBasis
NUM_POLYNOMIAL_CUTOFF = 5  # radial_embedding.cutoff_fn.p; PolynomialCutoff
RADIAL_TYPE = "bessel"  # type(radial_embedding.bessel_fn).__name__ == "BesselBasis"
DISTANCE_TRANSFORM = None  # radial_embedding has no distance_transform submodule

# --- spherical harmonics: extracted from model.spherical_harmonics.irreps_out ---
# irreps = 1x0e+1x1o+1x2e+1x3o -> max l = 3
MAX_ELL = 3

# --- interactions: extracted from model.interactions[i] ---
# interactions[0]: RealAgnosticInteractionBlock
# interactions[1]: RealAgnosticResidualInteractionBlock
# (mace.modules.blocks — must use these exact classes, not just the hyperparams)
INTERACTION_CLS_FIRST = "mace.modules.blocks.RealAgnosticInteractionBlock"
INTERACTION_CLS = "mace.modules.blocks.RealAgnosticResidualInteractionBlock"

# hidden_irreps constructor arg: stored on block as interactions[i].hidden_irreps
HIDDEN_IRREPS = "128x0e+128x1o"  # scalar part (node_feats_irreps) = 128x0e
# derived inside MACE.__init__ (not a constructor arg): sh_irreps x 128 features
INTERACTION_IRREPS = "128x0e+128x1o+128x2e+128x3o"  # == blocks' target_irreps

# --- products: extracted from model.products[i].symmetric_contractions ---
# products[0]: irreps 128x0e+128x1o+128x2e+128x3o -> 128x0e+128x1o, use_sc=False
# products[1]: irreps 128x0e+128x1o+128x2e+128x3o -> 128x0e,         use_sc=True
CORRELATION = 3  # contractions[0].correlation on both products

# --- node embedding: extracted from model.node_embedding ---
# LinearNodeEmbeddingBlock; linear.weight shape (1280,) = 128 x 10
NODE_EMBEDDING_IN = 10   # num_elements
NODE_EMBEDDING_OUT = 128  # == scalar channels of hidden_irreps

# --- readout: extracted from model.readouts ---
# readouts[0]: LinearReadoutBlock   (linear.weight (128,))
# readouts[1]: NonLinearReadoutBlock(linear_1 (2048,)=128x16, linear_2 (16,), gate=silu)
MLP_IRREPS = "16x0e"  # readouts[-1].hidden_irreps
GATE = "silu"  # mace default gate used by run_train.py (non_linearity acts=normalize2mom)

# --- normalization constant: extracted from interactions[0].avg_num_neighbors ---
AVG_NUM_NEIGHBORS = 18.41771125793457  # exact stored value

# --- radial weight MLP: extracted from conv_tp_weights.hs ---
# hs = [8, 64, 64, 64, 512] == [edge_feats_dim, *radial_MLP, out]
RADIAL_MLP = [64, 64, 64]

# --- atomic reference energies from checkpoint (shape (10,)) ---
# NOTE: checkpoint stores 1D [10]; repo model.py assigns [1,10] and squeezes on save.
# For from-scratch: pass zeros here; fit_atomic_references overwrites afterwards.
ATOMIC_ENERGIES_SHAPE = (10,)
CHECKPOINT_ATOMIC_ENERGIES = [
    -13.571965217590332, -1030.567138671875, -1486.375, -2043.9337158203125,
    -2715.318603515625, -9287.4072265625, -10834.484375, -12522.6494140625,
    -70045.28125, -8102.5244140625,
]

# --- scale/shift from the checkpoint (learned during MACE-OFF23 training) ---
# From-scratch model: start scale=1.0, shift=0.0; calibration handled per Part 4.
CHECKPOINT_SCALE = 1.088502287864685
CHECKPOINT_SHIFT = 0.0

# --- misc ---
NUM_PARAMS = 1428368  # sum(p.numel() for p in model.parameters())
