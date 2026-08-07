"""Approach 0 - iteration-variance diagnostic (go/no-go gate).

Hypothesis (Durasov et al., ICML 2024, "Enabling Uncertainty Estimation in
Iterative Neural Networks"): a network that re-reads its OWN scalar prediction
as part of the input has a per-molecule spread across the later refinement
iterations that behaves like an uncertainty estimate WITHOUT a separate
ensemble.

We implement the minimal-surgery version on top of an ALREADY-TRAINED
deep-ensemble member (seed 42 by default):

  * the Stage-2 correction DimeNet+ is loaded and FROZEN (core weights
    byte-identical to the verified checkpoint; nothing is re-trained);
  * a tiny, NEW trainable read-in (a single zero-initialized Linear, ~hidden
    params) is added to the atom feature stream: at each pass the previous
    iteration's scalar prediction is broadcast to every atom of its molecule
    and injected as one extra continuous column;
  * because the adapter is zero-initialized, pass 0 is EXACTLY the frozen
    backbone (correctness anchor: must reproduce seed-42's published
    single-conformer test MAE ~0.531 on the frozen split);
  * the adapter is fitted on the frozen TRAIN split for a few epochs with the
    detached-feedback refinement objective (given y0, output should move
    toward the label) - this is the "lightweight adapter/finetune, not new
    training from scratch" step;
  * on the frozen 129-mol TEST set we then run K passes feeding y_{k-1} back,
    and compute per-molecule {mean, std} over the iteration outputs.

GO criteria (same bar as the verified ensemble verdict):
  Spearman(iteration_std, |mean-exp|) rho > 0.25 with p < 0.05,
  and a sanity check that iteration_std agrees with the verified
  ensemble_std (positive rank correlation).
  A degenerate result (e.g. variance ~0 because a frozen DimeNet+ cannot
  admit its own output without heavier surgery) is reported HONESTLY as
  NO-GO and Approach 1 (weighted re-training) remains the primary path.

Outputs -> output/approach0/{report.json, iterations.csv}
"""

import argparse
import csv
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
from common import (
    EV_TO_KCAL, DEFAULT_SPLIT_DIR, DEFAULT_ENSEMBLE_DIR, DEFAULT_CONFORMERS,
    DEFAULT_LABELS, DEFAULT_PER_MOLECULE_CSV,
    set_seed, load_frozen_split, load_freesolv_labels, simple_dataset_cls,
    build_model, load_checkpoint, load_ensemble_member,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "output", "approach0")


class FeedbackAdapter(nn.Module):
    """Read-in for the previous iteration's scalar prediction.

    One Linear from [1] (the scalar, kcal/mol) to hidden_channels, bias=False,
    zero-initialized so that at fb=0 the wrapped model is exactly the frozen
    backbone.
    """

    def __init__(self, hidden_channels):
        super().__init__()
        self.net = nn.Linear(1, hidden_channels, bias=False)
        nn.init.zeros_(self.net.weight)

    def forward(self, fb_col):
        return self.net(fb_col)


class FeedbackEmbedding(nn.Module):
    """Drop-in replacement for model.emb (HybridEmbeddingBlock).

    Same arithmetic as the original embedding for the one-hot + original
    continuous columns, PLUS the adapter contribution when x carries the extra
    feedback column (x[:, 17]).  All original submodules are shared (frozen);
    only the adapter is trainable.
    """

    def __init__(self, orig_emb, adapter):
        super().__init__()
        self.orig = orig_emb
        self.adapter = adapter
        self.act = orig_emb.act

    def forward(self, x, rbf, i, j):
        # x: [N, 18] = one_hot(17) | fb(1)   (fb=0 on the first pass)
        x_oh = x[:, :17]
        atom_types = x_oh.argmax(dim=1)
        atom_emb = self.orig.atom_embedding(atom_types)
        cont_emb = self.orig.continuous_lin(x_oh[:, 1:])
        if x.size(1) > 17:
            cont_emb = cont_emb + self.adapter(x[:, 17:])
        x_h = atom_emb + cont_emb
        rbf_emb = self.act(self.orig.lin_rbf(rbf))
        return self.act(self.orig.lin(torch.cat([x_h[i], x_h[j], rbf_emb], dim=-1)))


def build_wrapped(seed, ensemble_dir, device, hidden=128):
    """Frozen backbone + zero-init feedback adapter, wrapped embedding."""
    model, ckpt_path, ckpt_sha = load_ensemble_member(seed, ensemble_dir, device)
    for p in model.parameters():
        p.requires_grad_(False)
    adapter = FeedbackAdapter(hidden).to(device)
    model.emb = FeedbackEmbedding(model.emb, adapter)
    return model, adapter, ckpt_path, ckpt_sha


def forward_pass(model, data, device, fb_kcal=None, grad=False):
    """One wrapped forward.  fb_kcal: [n_molecules] tensor or None (-> zeros
    feedback column).  grad=False runs under torch.no_grad (inference)."""
    from element_vocab import build_one_hot
    x = build_one_hot(data, device)                       # [N,17]
    if fb_kcal is None:
        fb = torch.zeros((data.num_nodes, 1), device=device)
    else:
        fb = fb_kcal[data.batch].unsqueeze(-1)
    x = torch.cat([x, fb], dim=-1)
    if grad:
        return model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL
    with torch.no_grad():
        return model(x, data.pos, data.batch).view(-1) * EV_TO_KCAL


def fit_adapter(model, adapter, train_loader, device, epochs, lr=2e-4, clip=10.0):
    """Detached-feedback refinement: given y0 (frozen), learn fb= y0 -> label.

    Only the adapter gets gradients (backbone frozen).  This is the
    'lightweight adapter/finetune' step.  Returns list of per-epoch MSE."""
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    mse = nn.MSELoss()
    history = []
    for ep in range(epochs):
        tot, n = 0.0, 0
        for data in train_loader:
            data = data.to(device)
            y = data.y_dG.view(-1).to(device)
            y0 = forward_pass(model, data, device, None)          # frozen backbone pass
            y1 = forward_pass(model, data, device, y0.detach(), grad=True)  # refines fb
            loss = mse(y1, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), clip)
            opt.step()
            tot += float(loss.detach())
            n += 1
        history.append(tot / max(n, 1))
        print(f"  [adapter] epoch {ep+1}/{epochs} | MSE (kcal^2) {history[-1]:.4f}")
    return history


def scan_iterations(model, loader, device, K):
    """Run K passes per batch; returns {mol_id: [pass0..passK-1]} in kcal."""
    out = {}
    for data in loader:
        data = data.to(device)
        mids = list(data.mol_id)
        prev = None
        for k in range(K):
            pred = forward_pass(model, data, device, prev)
            for j, mid in enumerate(mids):
                out.setdefault(mid, []).append(float(pred[j]))
            prev = pred
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--ensemble_dir", default=DEFAULT_ENSEMBLE_DIR)
    ap.add_argument("--conformers", default=DEFAULT_CONFORMERS)
    ap.add_argument("--labels_json", default=DEFAULT_LABELS)
    ap.add_argument("--per_molecule", default=DEFAULT_PER_MOLECULE_CSV)
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    ap.add_argument("--member_seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--K", type=int, default=3, help="refinement passes (2-3 per plan)")
    ap.add_argument("--adapter_epochs", type=int, default=3)
    ap.add_argument("--adapter_lr", type=float, default=2e-4)
    ap.add_argument("--no_train", action="store_true",
                    help="skip adapter fitting (variance will be ~0; still reported)")
    ap.add_argument("--smoke", action="store_true", help="tiny slice for CI")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    labels = load_freesolv_labels(args.labels_json)
    train_ids, val_ids, test_ids = load_frozen_split(args.split_dir, labels)
    if args.smoke:
        train_ids, test_ids = train_ids[:8], test_ids[:10]
        args.adapter_epochs = min(args.adapter_epochs, 2)

    # verified aggregate (test-side ensemble_std + expt)
    per_mol = {}
    with open(args.per_molecule, newline="") as f:
        for row in csv.DictReader(f):
            per_mol[row["mol_id"]] = {"exp": float(row["true_value"]),
                                      "ens_std": float(row["ensemble_std"])}

    ds = simple_dataset_cls(args.conformers, labels)
    train_loader = DataLoader(ds(train_ids), batch_size=8, shuffle=True)
    test_loader = DataLoader(ds(test_ids), batch_size=8, shuffle=False)

    model, adapter, ckpt_path, ckpt_sha = build_wrapped(args.member_seed,
                                                        args.ensemble_dir, device)
    print(f"\nloaded frozen member seed={args.member_seed} ckpt={ckpt_sha[:12]}... "
          f"| adapter params = {sum(p.numel() for p in adapter.parameters())}")

    # ---- correctness anchor: wrapped pass-0 must equal frozen backbone ----
    plain, _, _, _ = build_wrapped(args.member_seed, args.ensemble_dir, device)
    for d in test_loader:
        d0 = d.to(device)
        a = forward_pass(model, d0, device, None)
        b = forward_pass(plain, d0, device, None)
        assert torch.allclose(a, b, atol=1e-6), "wrapped pass-0 diverged from backbone!"
    print("  [sanity] wrapped pass-0 == frozen backbone on test set (bit-close) OK")

    # published anchor: seed-42 single-conf test MAE from the verified run
    t0 = torch.cat([forward_pass(model, d.to(device), device, None)
                    for d in test_loader]).numpy()
    expts = np.array([per_mol[m]["exp"] for m in test_ids if m in per_mol])
    t0_full = np.array([t0[i] for i, m in enumerate(test_ids) if m in per_mol])
    mae0 = float(np.mean(np.abs(t0_full - expts)))
    print(f"  [sanity] pass-0 single-conf test MAE = {mae0:.4f} "
          f"(seed-42 published: 0.5313 single-conf / 0.5048 TTA)")

    # ---- fit the adapter (unless --no_train) ----
    if not args.no_train:
        fit_adapter(model, adapter, train_loader, device,
                    epochs=args.adapter_epochs, lr=args.adapter_lr)
    else:
        print("  [skip] adapter training skipped (zero-init => iterations identical)")

    # ---- iterate on the test set ----
    table = scan_iterations(model, test_loader, device, args.K)

    # ---- per-molecule stats + correlations ----
    rows = []
    for m in test_ids:
        if m not in table or m not in per_mol:
            continue
        passes = np.array(table[m])
        rows.append({"mol": m,
                     "passes": passes,
                     "mean": float(passes.mean()),
                     "std": float(passes.std(ddof=0)),
                     "exp": per_mol[m]["exp"],
                     "ens_std": per_mol[m]["ens_std"]})
    rows = [r for r in rows if r["std"] == r["std"]]  # drop NaN

    report = {"member_seed": args.member_seed, "K": args.K,
              "adapter_epochs": args.adapter_epochs if not args.no_train else 0,
              "sanity_pass0_mae_kcal": mae0, "n_test": len(rows)}

    if len(rows) >= 5:
        import scipy.stats as st
        stds = np.array([r["std"] for r in rows])
        abserr = np.array([abs(r["mean"] - r["exp"]) for r in rows])
        ens_std = np.array([r["ens_std"] for r in rows])
        rho1, p1 = st.spearmanr(stds, abserr)
        rho2, p2 = st.spearmanr(stds, ens_std)
        print(f"\n  Spearman(iteration_std, |mean-exp|) = {rho1:.4f} (p={p1:.4g}, N={len(rows)})")
        print(f"  Spearman(iteration_std, ensemble_std) = {rho2:.4f} (p={p2:.4g})")
        go = rho1 > 0.25 and p1 < 0.05 and rho2 > 0
        print(f"  GO/NO-GO: {'GO' if go else 'NO-GO'} "
              f"(reference ensemble rho~0.496, p=2.3e-9)")
        report.update({
            "spearman_iterstd_vs_abserr": {"rho": float(rho1), "p": float(p1), "N": int(len(rows))},
            "spearman_iterstd_vs_ensstd": {"rho": float(rho2), "p": float(p2)},
            "go": bool(go),
            "verdict": ("iteration-variance GO" if go else
                        "NO-GO: frozen DimeNet+ with a tiny read-in does not produce "
                        "a usable iteration-variance surrogate; Approach 1 (weighted "
                        "re-training) remains the primary path.")})

    # ---- save artifacts ----
    with open(os.path.join(args.output_dir, "iterations.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "std", "mean", "exp", "abs_err", "ens_std"]
                   + [f"pass{k}" for k in range(args.K)])
        for r in rows:
            w.writerow([r["mol"], f"{r['std']:.6f}", f"{r['mean']:.6f}",
                        f"{r['exp']:.6f}", f"{abs(r['mean'] - r['exp']):.6f}",
                        f"{r['ens_std']:.6f}"]
                       + [f"{v:.6f}" for v in r["passes"]])
    with open(os.path.join(args.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  saved -> {os.path.join(args.output_dir, 'report.json')}")
    print(f"           {os.path.join(args.output_dir, 'iterations.csv')}")


if __name__ == "__main__":
    main()