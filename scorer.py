"""
scorer.py — External scorer for REINVENT4
|||||||||||||||||||||||||||||||||||||||||
Bridge between REINVENT4 (ExternalProcess) and a Chemprop v2 model (.ckpt).

REINVENT4 can call this script in two ways:

  Mode 1 — stdin (ExternalProcess during RL):
    echo "SMILES1\nSMILES2" | python scorer.py

  Mode 2 — CSV as argument (manual call / verification):
    python scorer.py <smiles_csv>

The script prints to stdout a CSV with columns "SMILES,score,pActivity_pred".

The score is the predicted pActivity transformed by a sigmoid:
    score = sigmoid(pActivity; low=6.0, high=9.0, k=0.5)
     score ≈ 0 for pActivity ≤ 5  (≥ 10 µM, inactive)
     score ≈ 0.5 for pActivity = 7.5 (≈ 30 nM)
     score ≈ 1 for pActivity ≥ 9  (≤ 1 nM, very potent)
"""

import sys
import math
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Checkpoint — absolute path relative to the script (works from any CWD)
_SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT  = str(_SCRIPT_DIR / "chemprop_model" / "MPNN_5ht2a.ckpt")

# Sigmoid parameters (calibrated against the model's real output range)
# pActivity dataset: min=4, mean=7.15, max=11, model RMSE=0.85
SIG_LOW  = 6.0   # pActivity where the score starts to rise (Ki = 1 µM)
SIG_MID  = 7.5   # inflection point (Ki ≈ 30 nM)  → score = 0.5
SIG_HIGH = 9.0   # pActivity where the score saturates   (Ki = 1 nM)
SIG_K    = 0.8   # slope; higher = more selective towards high potency

# ─────────────────────────────────────────────────────────────────────────────

def sigmoid_transform(x: float, mid: float = SIG_MID, k: float = SIG_K) -> float:
    """Standard sigmoid centered at `mid` with slope `k`."""
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - mid)))
    except (OverflowError, ValueError):
        return 0.0


def load_model(ckpt_path: str):
    """Loads the Chemprop v2 MPNN model from a Lightning checkpoint into CPU memory."""
    from chemprop import models
    model = models.MPNN.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    return model


def predict_pactivity(smiles_list: list, model) -> np.ndarray:
    """
    Predicts pActivity for a list of SMILES strings.

    Args:
        smiles_list (list): List of SMILES strings to evaluate.
        model: A loaded Chemprop MPNN model.

    Returns:
        np.ndarray: Array of predicted pActivity values. Invalid SMILES get NaN.
    """
    import os
    import logging
    import sys as _sys
    from chemprop import data, featurizers
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    # Silence Lightning — its Tips/GPU messages must not contaminate stdout
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    os.environ["PYTORCH_LIGHTNING_SUPPRESS_TIPS"] = "1"

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    datapoints, valid_idx = [], []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is not None:
            datapoints.append(data.MoleculeDatapoint.from_smi(smi))
            valid_idx.append(i)

    preds = np.full(len(smiles_list), np.nan)

    if not datapoints:
        return preds

    dset   = data.MoleculeDataset(datapoints, featurizer)
    loader = data.build_dataloader(dset, num_workers=0, shuffle=False)

    with torch.no_grad():
        import lightning.pytorch as pl

        # Redirect stdout → stderr during trainer.predict
        # so Lightning messages don't contaminate the CSV/JSON output
        _real_stdout = _sys.stdout
        _sys.stdout  = _sys.stderr
        try:
            trainer = pl.Trainer(
                accelerator="gpu", # cuda:0
                devices=1,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            raw = trainer.predict(model, dataloaders=loader)
        finally:
            _sys.stdout = _real_stdout  # always restore

    values = torch.cat(raw).squeeze().numpy()
    if values.ndim == 0:
        values = values.reshape(1)

    for idx, val in zip(valid_idx, values):
        preds[idx] = float(val)

    return preds


def main():
    """
    Entry point. Determines input mode, runs a single prediction pass,
    and formats the output accordingly (JSON for REINVENT4, CSV for manual use).
    """
    import json

    # ── Step 1: Parse input based on invocation mode ──────────────────────────
    if len(sys.argv) == 1:
        # Mode 1: REINVENT4 pipes SMILES via stdin, expects JSON back
        raw = sys.stdin.read().strip()
        if not raw:
            print(json.dumps({"score": [], "pActivity_pred": []}))
            return
        smiles_list = [s.strip() for s in raw.splitlines() if s.strip()]
        output_mode = "json"
        smiles_col  = None

    elif len(sys.argv) == 2:
        # Mode 2: Manual call with a CSV file, returns CSV
        df = pd.read_csv(sys.argv[1])
        smiles_col  = next(
            (c for c in df.columns if c.lower() == "smiles"),
            df.columns[0]
        )
        smiles_list = df[smiles_col].astype(str).tolist()
        output_mode = "csv"

    else:
        print("Usage: python scorer.py [<smiles_csv>]", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: Run prediction (single pass for both modes) ───────────────────
    model     = load_model(CHECKPOINT)
    pactivity = predict_pactivity(smiles_list, model)
    scores    = np.array([
        sigmoid_transform(v) if not np.isnan(v) else 0.0
        for v in pactivity
    ])

    # ── Step 3: Format and print output ───────────────────────────────────────
    if output_mode == "json":
        print(json.dumps({
            "payload": {
                "score":          [round(float(s), 6) for s in scores],
                "pActivity_pred": [round(float(v), 3) if not np.isnan(v) else None
                                   for v in pactivity],
            }
        }))
    else:
        out = pd.DataFrame({
            smiles_col:       smiles_list,
            "score":          scores.round(6),
            "pActivity_pred": np.round(pactivity, 3),
        })
        print(out.to_csv(index=False), end="")


if __name__ == "__main__":
    import traceback
    _log = _SCRIPT_DIR / "scorer_debug.log"
    try:
        main()
    except Exception:
        with open(_log, "w") as f:
            f.write(traceback.format_exc())
        sys.exit(1)
