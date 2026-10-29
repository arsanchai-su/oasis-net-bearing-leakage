# OASIS-Net: Order-Aware, Specimen-Invariant Scalogram Network for Bearing Fault Diagnosis

Code, protocols and results for the paper:

> A. Sukkuea, N. K. Tharu, P. Akkajit. *OASIS-Net: An Order-Aware, Specimen-Invariant Scalogram Network for Bearing Fault Diagnosis under Leakage-Controlled Evaluation.* Submitted to *Machine Learning and Knowledge Extraction* (MDPI), 2026.

## What is here

| Path | Contents |
|---|---|
| `notebooks/OASIS_Net_leakage_benchmark.ipynb` | Single Colab notebook: data download, leakage-ladder protocols, six signal-to-image encodings, ResNet-18 / Swin-T / CNN–Swin hybrid, WDCNN, gradient boosting, OASIS-Net and ablations, configuration search, noise test, statistics. Saved with the outputs of the run reported in the paper. |
| `analysis/analysis.py` | Recomputes every number quoted in the paper from `results/` into `numbers.json`. |
| `analysis/figures.py` | Redraws the paper figures (design, OASIS-Net, ladder, L0-vs-leakage-free, critical-difference, noise). |
| `analysis/fig_oasis_profile.py` | Draws Figure 10, the OASIS-Net profile on Paderborn. |
| `results/results.csv` | One row per trained run (configuration × protocol × fold × seed × test set × SNR). |
| `results/tables/` | All tables (CSV / Markdown / LaTeX), including the leakage audit. |
| `results/figs/` | Figures written by the notebook. |
| `results/oasis2_selection.csv` | 24-configuration search on held-out PU training bearings (secondary analysis). |
| `figures/` | Publication figures 1–10 of the paper (PNG at 600 dpi; vector PDF where available). |

## Leakage ladder

| Protocol | Data | Test set | Still shared with training |
|---|---|---|---|
| L0 random | CWRU, PU | random 30% of overlapped windows | overlapping samples, recording, specimen, condition |
| L1 temporal / repetition | CWRU, PU | last 30% of each recording / later measurements | recording or bearing, condition |
| L2 load-out | CWRU | the held-out motor load | specimen |
| L3 size-out / bearing | CWRU, PU | unseen fault size / unseen bearings | healthy CWRU bearing only / nothing |
| X condition | PU | unseen bearings under the other three settings | nothing |
| A2R | PU | real-damage bearings, trained on artificial damage | nothing |

## Reproducing

1. Open the notebook in Google Colab and choose **Runtime → Change runtime type → T4 GPU**.
2. Mount Google Drive when asked; every finished run is appended to `results.csv` on Drive, so an interrupted session resumes where it stopped.
3. Run all cells. The `T4_FAST` preset (top-1 encoding, 6 PU measurements, 10 epochs) is the configuration used in the paper.
4. To recompute the paper numbers and figures from the saved results:

```bash
pip install -r requirements.txt
mkdir -p out/figs
python analysis/analysis.py results out
python analysis/figures.py results out
```

## Data

- **CWRU**: Case Western Reserve University Bearing Data Center, <https://engineering.case.edu/bearingdatacenter>.
- **Paderborn (KAt)**: <https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter>, CC BY-NC 4.0 (please credit Lessmeier, Kimotho, Zimmer and Sextro, PHM Society European Conference 2016). The notebook downloads the archives from the Zenodo copy by J. van den Hoogen (2025, CC BY 4.0), <https://doi.org/10.5281/zenodo.15845309>, because the KAt server refused automated downloads. KA08 is missing from that copy and is excluded.

The datasets are not redistributed in this repository.

## Main result

Under random overlapped splits the six main configurations score 91–100% macro-F1; on unseen specimens and conditions it drops to 39–63%. On the eight leakage-free test blocks the STFT CNN–Swin hybrid ranks first (mean rank 1.75, Friedman p = 0.004) but is not significantly better than tuned gradient boosting on hand-crafted features. OASIS-Net reaches 51.9%, ahead of ResNet-18 and WDCNN, and is more noise-robust than the hybrid on unseen Paderborn bearings; its 54k-parameter order-only variant, OASIS-Lite, shows the smallest inflation and seed variance on Paderborn. A specimen-identification probe shows that the adversary leaves specimen identity largely intact.

## Licence

Code: MIT (see `LICENSE`). Results files: CC BY 4.0.
