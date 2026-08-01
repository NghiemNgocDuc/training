"""Random-init MACE-OFF23 'medium' clone for from-scratch training.

Same architecture as the loaded MACE-OFF23 'medium' checkpoint (mace 0.3.16),
but with random weights. Every constant is read from mace_off23_medium_arch_config.py
(extracted from the loaded checkpoint object on 2026-08-01), NOT from papers.

This module is additive: model.py (the pretrained path) is left untouched.
MACEFreeSolvScratch mirrors MACEFreeSolv.__init__ exactly, with a single
difference: base = build_scratch_mace() instead of load_mace_foundation().
"""

import numpy as np
import torch
from mace import modules
from mace.modules import ScaleShiftMACE

import e3nn.o3 as o3

from mace_off23_medium_arch_config import (
    R_MAX,
    NUM_RADIAL_BASIS,
    NUM_POLYNOMIAL_CUTOFF,
    MAX_ELL,
    NUM_INTERACTIONS,
    NUM_ELEMENTS,
    HIDDEN_IRREPS,
    MLP_IRREPS,
    AVG_NUM_NEIGHBORS,
    ATOMIC_NUMBERS,
    CORRELATION,
    RADIAL_MLP,
)
from mace.modules.lora import inject_LoRAs
from model import MACEFreeSolv, calibrate_output, fit_atomic_references


def build_scratch_mace(device="cpu"):
    """Randomly-initialized ScaleShiftMACE matching the MACE-OFF23 'medium'
    architecture. atomic_inter_scale/shift = 1.0/0.0 (no shift; the repo's
    fit_atomic_references + calibrate_output step runs afterwards, exactly as
    in the pretrained path)."""
    model = ScaleShiftMACE(
        r_max=R_MAX,
        num_bessel=NUM_RADIAL_BASIS,
        num_polynomial_cutoff=NUM_POLYNOMIAL_CUTOFF,
        max_ell=MAX_ELL,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=NUM_INTERACTIONS,
        num_elements=NUM_ELEMENTS,
        hidden_irreps=o3.Irreps(HIDDEN_IRREPS),
        MLP_irreps=o3.Irreps(MLP_IRREPS),
        atomic_energies=np.zeros(NUM_ELEMENTS),
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        atomic_numbers=ATOMIC_NUMBERS,
        correlation=CORRELATION,
        gate=modules.gate_dict["silu"],
        pair_repulsion=False,
        apply_cutoff=True,
        use_reduced_cg=False,
        use_so3=False,
        use_agnostic_product=False,
        use_last_readout_only=False,
        distance_transform="None",
        radial_MLP=RADIAL_MLP,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )
    return model.to(device)


class MACEFreeSolvScratch(MACEFreeSolv):
    """Same wrapper as MACEFreeSolv (fit_refs, calibration, LoRA, save/load)
    but built from random weights. Mirrors model.py:MACEFreeSolv.__init__ with
    the base-model construction line as the only difference."""

    def __init__(self, model_size="medium", device="cpu", fit_refs=True,
                 freeze_atomic_energies=False, target_mean=0.0, target_std=None,
                 use_lora=False, lora_rank=32, lora_alpha=2.0,
                 lora_unfreeze_readouts=True, lora_unfreeze_skip_tp=True,
                 fit_dataset=None, init_checkpoint=None):
        # Bypass MACEFreeSolv.__init__ (it loads the MACE-OFF23 foundation and
        # fits refs on FreeSolv); this path must never touch pretrained weights.
        torch.nn.Module.__init__(self)
        self.device = device
        self.model_size = model_size
        self.freeze_atomic_energies = freeze_atomic_energies
        self.target_mean = target_mean
        self.target_std = target_std
        self.use_lora = use_lora

        base = build_scratch_mace(device)
        base = base.float()
        for p in base.parameters():
            p.requires_grad_(True)

        if init_checkpoint is not None:
            state = torch.load(init_checkpoint, map_location=device, weights_only=True)
            if "atomic_energies_fn.atomic_energies" in state:
                expected_shape = base.atomic_energies_fn.atomic_energies.shape
                loaded = state["atomic_energies_fn.atomic_energies"]
                if loaded.shape != expected_shape:
                    state["atomic_energies_fn.atomic_energies"] = loaded.reshape(expected_shape)
            base.load_state_dict(state)
            print(f"  Initialized from Stage-A checkpoint: {init_checkpoint}")
            if fit_refs:
                print("  fit_refs disabled (checkpoint supplies atomic energies)")
            fit_refs = False

        if fit_refs:
            if fit_dataset is not None:
                ds = fit_dataset
            else:
                from data import MACEFreeSolvDataset
                ds = MACEFreeSolvDataset()
            ref_kcal = fit_atomic_references(ds)
            ref_ev = ref_kcal / 23.0605
            base.atomic_energies_fn.atomic_energies.data = ref_ev.unsqueeze(0).float()
            base.atomic_energies_fn.atomic_energies.requires_grad_(not self.freeze_atomic_energies)

            # Raw random-init output stats (post-refs, pre-training): mean/std
            # printed here feed the Part-4 calibration reasoning. For a fresh
            # model the refs dominate the output, so the mean/std are
            # ref-driven and the scale cap behavior mirrors the pretrained path.
            mean_e, std_e = calibrate_output(base, ds, batch_size=64, device=device)

            if target_std is None:
                target_std_ev = 1.0 / 23.0605
            else:
                target_std_ev = target_std / 23.0605
            scale_analytic = target_std_ev / max(std_e, 1e-8)
            scale = min(1.0, scale_analytic)
            shift = 0.0
            base.scale_shift.scale.data.fill_(scale)
            base.scale_shift.shift.data.fill_(shift)
            print(f"  Calibration: target_std={target_std_ev:.4f} eV, model_std={std_e:.4f} eV")
            if scale < scale_analytic:
                print(f"    scale={scale:.6f} (capped from {scale_analytic:.6f})")
                print(f"    WARNING: output will be ~{1/scale:.0f}x too small until scale_shift adapts")
            else:
                print(f"    scale={scale:.6f}")
            print(f"    shift=0.0 (per-node shift must be 0 — summed over atoms)")

        if use_lora:
            inject_LoRAs(base, rank=lora_rank, alpha=lora_alpha)
            lora_adapter = sum(p.numel() for p in base.parameters() if p.requires_grad)

            if not self.freeze_atomic_energies:
                base.atomic_energies_fn.atomic_energies.requires_grad_(True)
            base.scale_shift.scale.requires_grad_(True)
            base.scale_shift.shift.requires_grad_(True)

            if lora_unfreeze_skip_tp:
                n_skip = 0
                for name, p in base.named_parameters():
                    if "skip_tp" in name and "weight" in name:
                        p.requires_grad_(True)
                        n_skip += p.numel()

            if lora_unfreeze_readouts:
                n_read = 0
                for name, p in base.named_parameters():
                    if name.startswith("readouts") and "lora" not in name and "weight" in name:
                        p.requires_grad_(True)
                        n_read += p.numel()

            print(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}, adapter_params={lora_adapter:,}")
            if lora_unfreeze_skip_tp and n_skip > 0:
                print(f"  LoRA hybrid: unfroze skip_tp ({n_skip:,})")
            if lora_unfreeze_readouts and n_read > 0:
                print(f"  LoRA hybrid: unfroze readouts ({n_read:,})")

        self.model = base
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"MACEFreeSolvScratch: {n_trainable:,}/{n_total:,} trainable params ({100*n_trainable/n_total:.1f}%)")
        print("  NOTE: random init — same architecture as MACE-OFF23 'medium', no pretrained weights")
