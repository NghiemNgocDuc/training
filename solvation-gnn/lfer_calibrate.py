"""LFER calibration: fit dG_exp = a * dG_pred + b, evaluate on hold-out."""

import csv
import math
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="freesolv_predictions.csv")
    parser.add_argument("--output", default=None,
                        help="Save calibrated predictions CSV")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load predictions
    preds, expts = [], []
    with open(args.input) as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = float(row["dG_B_kcal"])
            e = float(row["dG_exp_kcal"])
            preds.append(p)
            expts.append(e)

    preds = np.array(preds)
    expts = np.array(expts)
    print(f"Loaded {len(preds)} molecules")

    # Split
    p_train, p_test, e_train, e_test = train_test_split(
        preds, expts, test_size=args.test_size, random_state=args.seed
    )

    # Fit linear regression
    lr = LinearRegression()
    lr.fit(p_train.reshape(-1, 1), e_train)
    a, b = float(lr.coef_[0]), float(lr.intercept_)

    # Evaluate raw
    raw_ae = np.abs(p_test - e_test)
    raw_se = (p_test - e_test) ** 2
    raw_mae = np.mean(raw_ae)
    raw_rmse = np.sqrt(np.mean(raw_se))

    # Evaluate calibrated
    cal_test = lr.predict(p_test.reshape(-1, 1))
    cal_ae = np.abs(cal_test - e_test)
    cal_se = (cal_test - e_test) ** 2
    cal_mae = np.mean(cal_ae)
    cal_rmse = np.sqrt(np.mean(cal_se))

    print(f"\nLFER calibration (train={len(p_train)}, test={len(p_test)}):")
    print(f"  dG_exp = {a:.4f} * dG_pred + {b:.4f}")
    print(f"  R² = {lr.score(p_train.reshape(-1, 1), e_train):.4f}")
    print()
    print(f"  {'':<20} {'Raw':>10} {'Calibrated':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12}")
    print(f"  {'MAE (kcal/mol)':<20} {raw_mae:>10.3f} {cal_mae:>12.3f}")
    print(f"  {'RMSE (kcal/mol)':<20} {raw_rmse:>10.3f} {cal_rmse:>12.3f}")

    # Save calibrated predictions if requested
    if args.output:
        all_preds = lr.predict(preds.reshape(-1, 1))
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dG_pred_kcal", "dG_lfer_kcal", "dG_exp_kcal"])
            for p, c, e in zip(preds, all_preds, expts):
                w.writerow([f"{p:.6f}", f"{c:.6f}", f"{e:.6f}"])
        print(f"\nSaved calibrated predictions to {args.output}")


if __name__ == "__main__":
    main()