"""RNG-leak audit: does ANY part of the instrumented eval / TTA path consume
global RNG state (python random, numpy, torch CPU/CUDA)?

Question answered empirically: snapshot every global RNG state, run the code
path, snapshot again, and diff. If every state is byte-identical after the
block, the block is RNG-neutral and CANNOT desync the train DataLoader shuffle
(which draws from torch's default generator via torch.randperm per epoch).

Also demonstrates that the shuffle itself DOES consume torch RNG (one randperm
per epoch of train iteration), confirming step-2 of the protocol: the original
dataloader shuffle relies on the torch default generator, so ANY extra torch
RNG consumption in the loop would desync it from a deterministic replay.

Usage: python audit_rng.py [--n_tta_mols 10]
"""

import argparse
import json
import os
import random
import sys

sys.stdout.reconfigure(line_buffering=True)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_deep_ensemble = os.path.dirname(_script_dir)
_freesolv = os.path.dirname(_deep_ensemble)
if _freesolv not in sys.path:
    sys.path.insert(0, _freesolv)

from deep_ensemble import (
    REPO_ROOT, set_seed, build_model, load_frozen_split, simple_dataset_cls,
    DEFAULT_SPLIT_DIR, DEFAULT_CORRECTION_CKPT, DEFAULT_CONFORMERS,
)


def snapshot():
    import numpy as np
    import torch
    st = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    return st


def states_equal(a, b):
    import numpy as np
    import torch
    eq = {}
    eq["py"] = a["py"] == b["py"]
    eq["np"] = np.array_equal(a["np"][1], b["np"][1]) and a["np"][:1] == b["np"][:1] and a["np"][2:] == b["np"][2:]
    eq["torch_cpu"] = torch.equal(a["torch_cpu"], b["torch_cpu"])
    if "torch_cuda" in a:
        eq["torch_cuda"] = all(x.equal(y) for x, y in zip(a["torch_cuda"], b["torch_cuda"]))
    return eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tta_mols", type=int, default=10)
    args = ap.parse_args()

    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from freesolv_dataset import download_freesolv_data, load_freesolv_labels
    from instrument_finetune import evaluate_loader_instrumented

    report = {}

    # ---- setup identical to instrument_finetune.main() ----
    json_path, _ = download_freesolv_data(os.path.join(REPO_ROOT, "Data", "FreeSolv"))
    all_labels = load_freesolv_labels(json_path)
    train_ids, val_ids, test_ids = load_frozen_split(DEFAULT_SPLIT_DIR, all_labels)

    set_seed(42)
    model = build_model(torch.device("cpu"))
    ckpt = torch.load(DEFAULT_CORRECTION_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)

    SimpleDataset = simple_dataset_cls(DEFAULT_CONFORMERS, all_labels)
    train_ds, val_ds, test_ds = SimpleDataset(train_ids), SimpleDataset(val_ids), SimpleDataset(test_ids)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # ---- CHECK 1: full val + test eval pass ----
    s0 = snapshot()
    v_mae, v_rmse, _ = evaluate_loader_instrumented(model, torch.device("cpu"), val_loader)
    t_mae, t_rmse, _ = evaluate_loader_instrumented(model, torch.device("cpu"), test_loader)
    s1 = snapshot()
    report["check1_eval_pass"] = {
        "val_mae_kcal": round(v_mae, 6), "test_mae_kcal": round(t_mae, 6),
        "rng_state_equal_after": states_equal(s0, s1),
    }
    print(f"[audit] CHECK 1 eval pass (val+test): val MAE {v_mae:.4f} | "
          f"RNG equal after: {report['check1_eval_pass']['rng_state_equal_after']}")

    # ---- CHECK 2: RDKit ETKDGv3 TTA conformer generation (as in conformer_average) ----
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    def _gen_confs(smiles, n):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        params.pruneRmsThresh = 0.5
        conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n, params=params)
        if not conf_ids:
            return None
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
        if props is None:
            return None
        try:
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
        except Exception:
            pass
        return [np.array(mol.GetConformer(i).GetPositions(), dtype=np.float64)
                for i in range(mol.GetNumConformers())]

    tta_ids = test_ids[: args.n_tta_mols]
    s0 = snapshot()
    confs_a = [_gen_confs(all_labels[m]["smiles"], 5) for m in tta_ids]
    s1 = snapshot()
    confs_b = [_gen_confs(all_labels[m]["smiles"], 5) for m in tta_ids]
    s2 = snapshot()

    def conf_arrays_equal(ca, cb):
        if ca is None or cb is None:
            return ca is None and cb is None
        if len(ca) != len(cb):
            return False
        for a, b in zip(ca, cb):
            if a.shape != b.shape or not np.array_equal(a, b):
                return False
        return True

    same_run = all(conf_arrays_equal(a, b) for a, b in zip(confs_a, confs_b))
    report["check2_rdkit_tta"] = {
        "n_mols": len(tta_ids),
        "rng_state_equal_after": states_equal(s0, s1),
        "deterministic_across_calls": bool(same_run),
    }
    print(f"[audit] CHECK 2 RDKit TTA ({len(tta_ids)} mols, seed 42): "
          f"RNG equal after: {report['check2_rdkit_tta']['rng_state_equal_after']} | "
          f"conformers deterministic: {same_run}")

    # ---- CHECK 3: train loader shuffle consumes torch default RNG (step-2 premise) ----
    rng_before = torch.get_rng_state().clone()
    it = iter(train_loader)
    next(it)
    rng_after_first_batch = torch.get_rng_state().clone()
    next(it)
    rng_after_second_batch = torch.get_rng_state().clone()
    consumes = not torch.equal(rng_before, rng_after_first_batch)
    consumes_per_epoch_only = torch.equal(rng_after_first_batch, rng_after_second_batch)
    report["check3_train_loader_rng"] = {
        "shuffle_draws_from_torch_default_generator": bool(consumes),
        "drawn_once_per_epoch_iteration": bool(consumes_per_epoch_only),
    }
    print(f"[audit] CHECK 3 train loader: consumes torch default RNG at epoch start: "
          f"{consumes} | only at iteration start: {consumes_per_epoch_only}")

    # ---- CHECK 4: WHY the eval pass consumed torch RNG: iter() itself draws ----
    # torch's BaseDataLoaderIter.__init__ does
    #   self._base_seed = torch.empty((), dtype=torch.int64).random_()
    # i.e. EVERY DataLoader iterator creation consumes ONE draw from the torch
    # default generator, regardless of shuffle (verified for plain torch and
    # torch_geometric, shuffle=False and True). So each `for data in loader`
    # in an eval pass advances the shared stream: the original per-epoch loop
    # draws iter(train)+iter(val); the instrumented loop adds iter(test), plus
    # 2 extra iters at epoch 0 -> every epoch's shuffle starts from a shifted
    # stream. This is the leak: added eval passes change the TRAIN order.
    import torch.utils.data as tud
    from torch_geometric.data import Data as TGData

    class _DS(tud.Dataset):
        def __len__(self):
            return 8
        def __getitem__(self, i):
            return torch.tensor([float(i)])

    class _TGDS(tud.Dataset):
        def __len__(self):
            return 8
        def __getitem__(self, i):
            return TGData(z=torch.tensor([1]), pos=torch.tensor([[0.0, 0.0, 0.0]]),
                          y_dG=torch.tensor([0.0]))

    draws = {}
    for name, ldr in (
        ("plain_torch_shuffle_false", tud.DataLoader(_DS(), batch_size=2, shuffle=False)),
        ("plain_torch_shuffle_true", tud.DataLoader(_DS(), batch_size=2, shuffle=True)),
        ("torch_geometric_shuffle_false",
         DataLoader(_TGDS(), batch_size=2, shuffle=False)),
        ("torch_geometric_shuffle_true",
         DataLoader(_TGDS(), batch_size=2, shuffle=True)),
    ):
        torch.manual_seed(99)
        s = torch.get_rng_state().clone()
        iter(ldr)
        draws[name] = not torch.equal(s, torch.get_rng_state())
    report["check4_iter_consumes_torch_rng"] = {
        k: bool(v) for k, v in draws.items()}
    print(f"[audit] CHECK 4 iter(loader) draws torch default RNG: {draws}")

    out = os.path.join(_script_dir, "rng_audit_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[audit] report -> {out}")


if __name__ == "__main__":
    main()
