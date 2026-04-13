"""
score_5ht2a.py — Scorer externo para REINVENT4
================================================
Puente entre REINVENT4 (ExternalProcess) y un modelo Chemprop v2 (.ckpt).

REINVENT4 puede llamar este script de dos formas:

  Modo 1 — stdin (ExternalProcess en RL):
    echo "SMILES1\nSMILES2" | python score_5ht2a.py

  Modo 2 — CSV como argumento (llamada manual / verificación):
    python score_5ht2a.py <smiles_csv>

El script imprime en stdout un CSV con columnas "SMILES,score,pActivity_pred".

El score es la pActivity predicha transformada por una sigmoide:
    score = sigmoid(pActivity; low=6.0, high=9.0, k=0.5)
    → score ≈ 0 para pActivity ≤ 5  (Ki ≥ 10 µM, inactivo)
    → score ≈ 0.5 para pActivity = 7.5 (Ki ≈ 30 nM)
    → score ≈ 1 para pActivity ≥ 9  (Ki ≤ 1 nM, muy potente)
"""

import sys
import math
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Checkpoint — path absoluto relativo al script (funciona desde cualquier CWD)
_SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT  = str(_SCRIPT_DIR / "chemprop_model" / "best-epoch=28-val_loss=0.4784.ckpt")

# Parámetros de la sigmoide (calibrados contra el rango real del modelo)
# pActivity dataset: min=4, media=7.15, max=11, RMSE modelo=0.85
SIG_LOW  = 6.0   # pActivity donde el score empieza a subir (Ki = 1 µM)
SIG_MID  = 7.5   # punto de inflexión (Ki ≈ 30 nM)  → score = 0.5
SIG_HIGH = 9.0   # pActivity donde el score satura   (Ki = 1 nM)
SIG_K    = 0.8   # pendiente; más alto = más selectivo hacia alta potencia

# ─────────────────────────────────────────────────────────────────────────────

def sigmoid_transform(x: float, mid: float = SIG_MID, k: float = SIG_K) -> float:
    """Sigmoide estándar centrada en `mid` con pendiente `k`."""
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - mid)))
    except (OverflowError, ValueError):
        return 0.0


def load_model(ckpt_path: str):
    """Carga el modelo chemprop v2 desde el checkpoint de Lightning."""
    from chemprop import models
    model = models.MPNN.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    return model


def predict_pactivity(smiles_list: list, model) -> np.ndarray:
    """Predice pActivity para una lista de SMILES. Devuelve NaN para inválidos."""
    import os
    import logging
    import sys as _sys
    from chemprop import data, featurizers
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    # Silenciar Lightning — sus mensajes de Tips/GPU no deben contaminar stdout
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

        # Redirigir stdout → stderr durante trainer.predict
        # para que los mensajes de Lightning no contaminen el CSV de salida
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
            _sys.stdout = _real_stdout  # restaurar siempre

    values = torch.cat(raw).squeeze().numpy()
    if values.ndim == 0:
        values = values.reshape(1)

    for idx, val in zip(valid_idx, values):
        preds[idx] = float(val)

    return preds


def main():
    import json

    # ── Modo 1: REINVENT4 — stdin + JSON output ───────────────────────────────
    if len(sys.argv) == 1:
        raw = sys.stdin.read().strip()
        if not raw:
            print(json.dumps({"score": [], "pActivity_pred": []}))
            return
        smiles_list = [s.strip() for s in raw.splitlines() if s.strip()]

        model     = load_model(CHECKPOINT)
        pactivity = predict_pactivity(smiles_list, model)
        scores    = np.array([
            sigmoid_transform(v) if not np.isnan(v) else 0.0
            for v in pactivity
        ])

        print(json.dumps({
            "payload": {
                "score":          [round(float(s), 6) for s in scores],
                "pActivity_pred": [round(float(v), 3) if not np.isnan(v) else None
                                   for v in pactivity],
            }
        }))

    # ── Modo 2: llamada manual — CSV input + CSV output ───────────────────────
    elif len(sys.argv) == 2:
        df = pd.read_csv(sys.argv[1])
        smiles_col  = next(
            (c for c in df.columns if c.lower() == "smiles"),
            df.columns[0]
        )
        smiles_list = df[smiles_col].astype(str).tolist()

        model     = load_model(CHECKPOINT)
        pactivity = predict_pactivity(smiles_list, model)
        scores    = np.array([
            sigmoid_transform(v) if not np.isnan(v) else 0.0
            for v in pactivity
        ])

        out = pd.DataFrame({
            smiles_col:       smiles_list,
            "score":          scores.round(6),
            "pActivity_pred": np.round(pactivity, 3),
        })
        print(out.to_csv(index=False), end="")

    else:
        print("Uso: python score_5ht2a.py [<smiles_csv>]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import traceback
    _log = _SCRIPT_DIR / "scorer_debug.log"
    try:
        main()
    except Exception:
        with open(_log, "w") as f:
            f.write(traceback.format_exc())
        sys.exit(1)
