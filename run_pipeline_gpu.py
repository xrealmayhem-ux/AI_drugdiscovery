#!/usr/bin/env python3
"""
run_pipeline_gpu.py — Staged learning (GPU)
=============================================================
De Novo generation of molecules for 5-HT2A receptor using Reinvent 4 and Chemprop as scorer.
"""

import random
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# configuration
# ============================================================
CHEMPROP_PYTHON_PATH = "/home/sdageri/miniconda3/envs/chemprop/bin/python"
DEVICE = "cuda"
BATCH_SIZE = 256
S1_STEPS = 500
TL_EPOCHS = 20
S2_STEPS = 1000
# ============================================================

OUT_DIR = BASE_DIR / "results_gpu"
OUT_DIR.mkdir(exist_ok=True)

PRIOR_FILE = BASE_DIR / "reinvent.prior"
TL_SMILES_FILE = BASE_DIR / "tl_smiles.smi"
SCORER_SCRIPT = BASE_DIR / "scorer.py"

# logging function
def log(msg):
    print(f"[run_pipeline_gpu] {msg}", flush=True)

# run step function
def run_step(name, cmd, check_file=None):
    log(f"Starting: {name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"ERROR in {name}")
        log(result.stderr[-1000:] if result.stderr else "")
        sys.exit(1)
    if check_file and not Path(check_file).exists():
        log(f"ERROR: {name} did not generate {check_file}")
        sys.exit(1)
    log(f"OK: {name}")
    return result

# prepare smiles for transfer learning (split smiles in train and validation)
def prepare_smiles():
    log("Step 1: Preparing data for Transfer Learning")
    all_smiles = TL_SMILES_FILE.read_text().strip().splitlines()
    all_smiles = [s.split()[0] for s in all_smiles if s.strip()]
    n_train = int(len(all_smiles) * 0.8)
    tl_train = all_smiles[:n_train]
    tl_val = all_smiles[n_train:]
    TL_TRAIN = OUT_DIR / "tl_train.smi"
    TL_VAL = OUT_DIR / "tl_val.smi"
    TL_TRAIN.write_text("\n".join(tl_train))
    TL_VAL.write_text("\n".join(tl_val))
    log(f"  Train: {len(tl_train)} | Val: {len(tl_val)}")
    return TL_TRAIN, TL_VAL

# stage 1 toml configuration (Reinforcement learning 1)
def stage1_toml(prior, out):
    return f"""\
run_type = "staged_learning"
device   = "{DEVICE}"
tb_logdir       = "{out}/tb_stage1"
json_out_config = "{out}/_stage1.json"

[parameters]
prior_file         = "{prior}"
agent_file         = "{prior}"
summary_csv_prefix = "{out}/stage1"
batch_size         = {BATCH_SIZE}
use_checkpoint     = false

[learning_strategy]
type  = "dap"
sigma = 128
rate  = 0.0001

[[stage]]
max_score  = 1.0
max_steps  = {S1_STEPS}
chkpt_file = "{out}/stage1.chkpt"

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Alerts"
params.smarts = ["[*;r8]","[*;r9]","[*;r10]","[*;r11]","[*;r12]",
    "[*;r13]","[*;r14]","[*;r15]","[*;r16]","[*;r17]",
    "[#8][#8]","[#6;+]","[#16][#16]","[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]","C#C","C(=[O,S])[O,S]"]

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name   = "QED"
weight = 0.6

[[stage.scoring.component]]
[stage.scoring.component.TPSA]
[[stage.scoring.component.TPSA.endpoint]]
name   = "TPSA"
weight = 0.4
transform.type = "reverse_sigmoid"
transform.high = 90.0
transform.low  = 0.0
transform.k    = 0.5
"""

# transfer learning toml configuration
def tl_toml(input_model, output_model, train_smiles, val_smiles, out):
    return f"""\
run_type  = "transfer_learning"
device    = "{DEVICE}"
tb_logdir = "{out}/tb_tl"

[parameters]
num_epochs             = {TL_EPOCHS}
save_every_n_epochs    = 1
batch_size             = {BATCH_SIZE}
sample_batch_size      = 500

input_model_file       = "{input_model}"
output_model_file      = "{output_model}"
smiles_file            = "{train_smiles}"
validation_smiles_file = "{val_smiles}"

standardize_smiles   = true
randomize_smiles     = true
randomize_all_smiles = false
internal_diversity   = true
"""

# stage 2 toml configuration (Reinforcement learning 2) (chemprop)
def stage2_toml(prior, agent, scorer, chemprop_py, out):
    return f"""\
run_type = "staged_learning"
device   = "{DEVICE}"
tb_logdir       = "{out}/tb_stage2"
json_out_config = "{out}/_stage2.json"

[parameters]
prior_file         = "{prior}"
agent_file         = "{agent}"
summary_csv_prefix = "{out}/stage2"
batch_size         = {BATCH_SIZE}
use_checkpoint     = false

[learning_strategy]
type  = "dap"
sigma = 128
rate  = 0.0001

[diversity_filter]
type        = "IdenticalMurckoScaffold"
bucket_size = 10
minscore    = 0.6

[inception]
smiles_file = ""
memory_size = 50
sample_size = 10

[[stage]]
max_score  = 1.0
max_steps  = {S2_STEPS}
chkpt_file = "{out}/stage2.chkpt"

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name   = "ChemProp_5HT2A"
weight = 0.5
params.executable = "{chemprop_py}"
params.args       = "{scorer}"
params.property = "score"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Alerts"
params.smarts = ["[*;r8]","[*;r9]","[*;r10]","[*;r11]","[*;r12]",
    "[*;r13]","[*;r14]","[*;r15]","[*;r16]","[*;r17]",
    "[#8][#8]","[#6;+]","[#16][#16]","[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]","C#C","C(=[O,S])[O,S]"]

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name   = "QED"
weight = 0.3

[[stage.scoring.component]]
[stage.scoring.component.TPSA]
[[stage.scoring.component.TPSA.endpoint]]
name   = "TPSA"
weight = 0.2
transform.type = "reverse_sigmoid"
transform.high = 90.0
transform.low  = 0.0
transform.k    = 0.5
"""

# main function
def main():
    log(f"Start — Output: {OUT_DIR}")
    log(f"Device: {DEVICE} | Batch: {BATCH_SIZE}")

    tl_train, tl_val = prepare_smiles()

    log("Step 2: Stage 1 RL")
    shutil.rmtree(OUT_DIR / "tb_stage1", ignore_errors=True)
    s1_cfg = OUT_DIR / "stage1.toml"
    s1_cfg.write_text(stage1_toml(PRIOR_FILE, OUT_DIR))
    run_step("Stage 1", ["reinvent", "-l", str(OUT_DIR / "stage1.log"), str(s1_cfg)],
             check_file=OUT_DIR / "stage1.chkpt")

    log("Step 3: Transfer Learning")
    shutil.rmtree(OUT_DIR / "tb_tl", ignore_errors=True)
    tl_output = OUT_DIR / "tl.model"
    tl_cfg = OUT_DIR / "tl.toml"
    tl_cfg.write_text(tl_toml(OUT_DIR / "stage1.chkpt", tl_output, tl_train, tl_val, OUT_DIR))
    run_step("Transfer Learning", ["reinvent", "-l", str(OUT_DIR / "tl.log"), str(tl_cfg)])
    tl_ckpts = sorted(OUT_DIR.glob("tl.model.*.chkpt"))
    if not tl_ckpts:
        log("ERROR: No se generaron checkpoints de TL")
        sys.exit(1)
    tl_model = tl_ckpts[-1]
    log(f"  Usando: {tl_model.name}")

    log("Step 4: Stage 2 RL con ChemProp")
    shutil.rmtree(OUT_DIR / "tb_stage2", ignore_errors=True)
    s2_cfg = OUT_DIR / "stage2.toml"
    s2_cfg.write_text(stage2_toml(
        PRIOR_FILE, tl_model,
        SCORER_SCRIPT.resolve(), CHEMPROP_PYTHON_PATH, OUT_DIR
    ))
    run_step("Stage 2", ["reinvent", "-l", str(OUT_DIR / "stage2.log"), str(s2_cfg)],
             check_file=OUT_DIR / "stage2.chkpt")

    csv_files = list(OUT_DIR.glob("stage2_*.csv"))
    if csv_files:
        log(f"Pipeline completed. CSVs generated: {len(csv_files)}")
    else:
        log("WARNING: No CSVs generated in Stage 2")
    log("END.")


if __name__ == "__main__":
    main()
