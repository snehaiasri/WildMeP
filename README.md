# WiMeP (Streamlit version)

Prediction of DNA 5mC sites in two wild grasses, *Oryza rufipogon* and *Aegilops tauschii*, as a Streamlit app.
Based on [github.com/diprosinha/WiMeP](https://github.com/diprosinha/WiMeP). Uses Python 3.12.

## Run locally
```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/streamlit run app.py        # http://localhost:8501
```
The pinned `torch` wheel is for Linux x86-64. On another platform, replace that line in `requirements.txt` with `torch==2.14.1`.

## Deploy on Streamlit Community Cloud
1. Install [Git LFS](https://git-lfs.com) and push this folder to its own GitHub repository:
   ```
   git init -b main
   git lfs install --local
   git add .
   git commit -m "WiMeP Streamlit app"
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. At [share.streamlit.io](https://share.streamlit.io), create an app with that repository and `app.py` as the main file.
3. Under **Advanced settings**, choose Python **3.12**.

`requirements.txt` installs the CPU-only Python packages and `packages.txt` installs `libgomp1`, which LightGBM needs.

### Model files
The large model files (about 1 GB) are stored with Git LFS. If a host clones the repository without them, the app downloads
each missing file on first use and verifies its SHA-256. The source is the upstream repository's LFS storage; set
`WIMEP_LFS_BASE` to use another location.

### Settings
Set these as environment variables, or as top-level entries in the Community Cloud **Secrets** box.

| Setting | Default | |
|---|---|---|
| `WIMEP_MAX_BP` | `5000` | maximum bp per web job |
| `WIMEP_LFS_BASE` | upstream repository | where missing LFS files are downloaded from |

## Differences from the original web server
- **One species in memory at a time.** Community Cloud allows about 2.7 GB. Changing species, or choosing both, reloads the models.
- **No Random Forest.** It used over 1 GB of memory with an ensemble weight of 1.5e-9 or less. The `RF` output column is gone; on the
  1,000-site example sets, `prob_5mC` and the calls are identical to the full model at the reported four decimals.
  `slim_models.py` records how `ml_models_noRF.joblib` was made.
- **Lower web limit:** 5,000 bp rather than 20,000, because jobs run one at a time and 20,000 bp takes about 18 minutes on 2 CPU cores.
- **No `POST /predict` JSON endpoint.** Streamlit does not serve custom API routes; use the command line for scripted runs.
- **Submitted text is never treated as a file path.**

## Interface
- Top navigation bar with the logo on the left and the menu on the right; light, dark and system themes from the ⋮ menu (colours are in `.streamlit/config.toml`).
- The threshold slider, table filters and charts update without re-running the models.
- With both species selected, a **Compare models** tab shows where the two models agree.

## Command line
```
python wimep.py predict --species rufipogon --input example/Oryza_rufipogon_region.fasta --out region_predictions.csv
```
Options: `--mode auto|site|scan` and `--threshold` (default 0.5).

## Limitations
- Negatives come from coding sequences, so predictions partly reflect coding vs non-coding sequence composition. Scores are most reliable for genomic sequence of the matching species; unusual sequences (e.g. low-complexity repeats) can receive high scores.
- Predictions are not experimental measurements and must not be used as methylation labels.
- No 6mA prediction: no experimental 6mA data exist for these species.

## Authors
Dipro Sinha, Subham Ghosh, Abhik Sarkar, Sougata Bhattacharjee, Sayanti Guha Majumdar, Himanshushekhar Chaurasia, Sneha Murmu
