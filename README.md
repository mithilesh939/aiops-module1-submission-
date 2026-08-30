# aiops-module1-submission-

# AIOps Module 1 — Experiment Management & Reproducibility

**Name:** Mithilesh  
**Roll:** DA24B047  
**Repo:** https://github.com/mithilesh939/aiops-module1-submission-

---

## Submission Structure

```
aiops-module1-submission-/
├── solution.pdf          # Write-up for Q1, Q2 analysis, Q3 proof, Q4 note
├── train.py              # Q2: MLP training script with MLflow logging
├── q2_images.zip         # Q2: MLflow comparison screenshot
├── q3_images.zip         # Q3: Terminal screenshots — v1, v2, rollback proof
└── README.md             # This file
```

---

## Question 1 — Technical Debt Diagnosis

**Answer:** `solution.pdf` → Section Q1

- **(a)** Entanglement (CACE) — delivery time rounding cascaded to restaurant feature
- **(b)** Undeclared Consumers — marketing team silently reading model output
- **(c)** Configuration & Glue-Code Debt — 14 undocumented shell scripts
- **Mitigation:** DVC pipeline (`dvc.yaml` + `params.yaml`) replacing shell scripts

---

## Question 2 — MLflow Experiment Comparison

**Code:** `train.py` | **Screenshots:** `q2_images.zip` | **Analysis:** `solution.pdf` → Section Q2

### Run it
```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns
# second terminal
source myenv/bin/activate && python train.py
# open http://localhost:5000 → mnist-mlp-experiments → select all → Compare
```

### Results

| Run | lr | batch | val\_acc | test\_acc |
|-----|----|-------|----------|-----------|
| mlp-baseline | 0.001 | 32 | 0.962 | 0.961 |
| mlp-lr-0.01 | 0.01 | 32 | 0.972 | 0.958 |
| mlp-lr-0.0001 | 0.0001 | 32 | 0.948 | 0.956 |
| **mlp-batch-16** | **0.001** | **16** | **0.976** | **0.972** |
| mlp-batch-128 | 0.001 | 128 | 0.922 | 0.928 |
| mlp-deeper | 0.001 | 32 | 0.965 | 0.963 |

**Best run:** `mlp-batch-16` (batch=16, lr=0.001)

---

## Question 3 — DVC Data Versioning & Rollback

**Screenshots:** `q3_images.zip` | **Proof:** `solution.pdf` → Section Q3  
**SSH Remote Partner:** Navadeep

### Steps
```bash
# v1 — 1800 rows
dvc add filenames.csv && dvc push
git add filenames.csv.dvc && git commit -m "v1: 1800 rows" && git tag v1.0

# v2 — 2801 rows
# (appended 1001 new rows)
dvc add filenames.csv && dvc push
git add filenames.csv.dvc && git commit -m "v2: 2801 rows" && git tag v2.0

# Rollback to v1
git checkout v1.0
dvc checkout
wc -l filenames.csv    # → 1801 (confirmed)
```

Terminal output proving 1801 lines after rollback is in `q3_images.zip`.

---

## Question 4 — Capstone Reproducibility

**Role:** Partner B | **Partner A:** Pulkit (DA24B047)  
**Results:** `solution.pdf` → Section Q4

### Protocol (Partner B)
```bash
git clone <pulkit-repo>
git checkout <commit-sha>
dvc checkout
conda env create -f environment.yml
python train.py
```

### Result
| Metric | Partner A | Partner B |
|--------|-----------|-----------|
| accuracy | 1.0000 | 1.0000 |
| f1\_macro | 1.0000 | 1.0000 |

Metrics matched exactly (tolerance ±0.001). Note logged in MLflow run by Partner B.  
Screenshot of MLflow model page (accuracy=1, f1=1) is in `solution.pdf`.

---

## Screen Recordings

> **Google Drive:** `[add your drive link here]`

Recordings include: Q2 MLflow demo, Q3 rollback demo, Q4 reproduction run.

---

## Environment

```
Python  3.12
MLflow  3.15.2
DVC     3.67.1
```
