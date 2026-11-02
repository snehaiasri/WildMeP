#!/usr/bin/env python3
"""WiMeP: prediction of DNA 5mC sites in Oryza rufipogon and Aegilops tauschii.

Each species has its own model: an ensemble of XGBoost, LightGBM (on selected handcrafted +
DNABERT features), fine-tuned DNABERT-6 and a DNABERT-BiLSTM. 5mC = CG, CHG and CHH methylation as one class.

Input: FASTA (or CSV with a `sequence` column).
  site mode  every sequence is a 41-bp window with the candidate C at position 21
  scan mode  sequences of >= 41 bp are cut into overlapping 41-bp windows (step 1 bp); every window centred on a
             cytosine (C, or G = C on the opposite strand) is scored
  auto       site mode if all sequences are 41 bp with C at position 21, otherwise scan mode

usage:
  python wimep.py predict --species rufipogon --input seqs.fasta --out predictions.csv
  streamlit run app.py                                     # web interface
"""
import argparse, functools, io, json, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402

SPECIES = {"rufipogon": "Oryza_rufipogon", "tauschii": "Aegilops_tauschii"}
BASE_MODEL = os.path.join(HERE, "models", "dnabert_base")  # frozen zhihan1996/DNA_bert_6 (embeddings)
MODELS = ["XGBoost", "LightGBM", "DNABERT", "DNABERT-BiLSTM"]  # Random Forest omitted (weight ~1e-9, over 1 GB of memory): see slim_models.py
HALF, COMP = 20, str.maketrans("ACGTN", "TGCAN")
MAX_WEB_BP = int(os.environ.get("WIMEP_MAX_BP", 5000))
LFS_BASE = os.environ.get("WIMEP_LFS_BASE", "https://media.githubusercontent.com/media/diprosinha/WiMeP/main")


def kmers(s, k=6):
    return " ".join(s[i:i + k] for i in range(len(s) - k + 1))


def context(w):
    return "CG" if w[HALF + 1] == "G" else "CHG" if w[HALF + 2] == "G" else "CHH"


# ---------------------------------------------------------------- model files
def ensure_files(*paths):
    """Replace Git LFS pointer files with the real files (for hosts that clone without LFS); checks the SHA-256."""
    files = [os.path.join(d, f) for p in paths for d, _, fs in os.walk(p) for f in fs] + [p for p in paths if os.path.isfile(p)]
    for path in files:
        if os.path.getsize(path) > 512: continue
        with open(path, "rb") as fh: head = fh.read()
        if not head.startswith(b"version https://git-lfs"): continue
        import hashlib, urllib.request
        oid = re.search(rb"oid sha256:([0-9a-f]{64})", head).group(1).decode()
        url = f"{LFS_BASE}/{os.path.relpath(path, HERE).replace(os.sep, '/')}"
        print(f"downloading {url}", flush=True)
        sha, tmp = hashlib.sha256(), path + ".part"
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
            while chunk := r.read(1 << 20):
                sha.update(chunk); out.write(chunk)
        if sha.hexdigest() != oid:
            os.remove(tmp); raise RuntimeError(f"checksum mismatch for {url}")
        os.replace(tmp, path)


# ---------------------------------------------------------------- input
def read_input(text_or_path):
    """Return DataFrame(seq_id, sequence) from a FASTA/CSV path or raw FASTA / plain-sequence text (command line only)."""
    if os.path.exists(text_or_path):
        if text_or_path.endswith((".csv", ".csv.gz")):
            df = pd.read_csv(text_or_path, dtype=str)
            if "seq_id" not in df: df.insert(0, "seq_id", [f"seq{i + 1}" for i in range(len(df))])
            return df[["seq_id", "sequence"]]
        return parse_fasta(open(text_or_path).read())
    return parse_fasta(text_or_path)


def parse_fasta(text):
    """Parse FASTA or plain-sequence text. Use this, not read_input, for text submitted by web users: read_input
    would open a server file if the submitted text happened to be a valid path."""
    ids, seqs = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith(">"): ids.append(line[1:].split()[0] or f"seq{len(ids) + 1}"); seqs.append("")
        else:
            if not ids: ids.append("seq1"); seqs.append("")
            seqs[-1] += re.sub(r"\s", "", line)
    return pd.DataFrame(dict(seq_id=ids, sequence=[s.upper().replace("U", "T") for s in seqs]))


def windows(df, mode="auto"):
    """41-bp windows centred on a cytosine.

    site mode: each sequence is one 41-bp window with C at position 21.
    scan mode: the sequence (>= 41 bp) is cut into overlapping 41-bp windows with a step of 1 bp; every window whose
    centre is C (cytosine on the given strand) or G (cytosine on the opposite strand, window reverse-complemented)
    is scored. Columns: seq_id, position (1-based centre), strand, context, window_start, window_end, window."""
    is_site = (df.sequence.str.len() == 41) & (df.sequence.str[HALF] == "C")
    if mode == "auto": mode = "site" if is_site.all() else "scan"
    rows = []
    if mode == "site":
        for r in df.itertuples():
            if len(r.sequence) != 41 or r.sequence[HALF] != "C":
                raise ValueError(f"{r.seq_id}: each sequence must be 41 bp with C at position 21 (got {len(r.sequence)} bp)")
            if re.fullmatch("[ACGT]+", r.sequence):
                rows.append((r.seq_id, 21, "+", context(r.sequence), 1, 41, r.sequence))
    else:
        for r in df.itertuples():
            s = r.sequence
            for i in range(HALF, len(s) - HALF):
                if s[i] not in "CG": continue
                w = s[i - HALF:i + HALF + 1]
                if s[i] == "G": w = w.translate(COMP)[::-1]
                if re.fullmatch("[ACGT]+", w):
                    rows.append((r.seq_id, i + 1, "+" if s[i] == "C" else "-", context(w), i - HALF + 1, i + HALF + 1, w))
    return pd.DataFrame(rows, columns=["seq_id", "position", "strand", "context", "window_start", "window_end", "window"]), mode


# ---------------------------------------------------------------- models
@functools.lru_cache(None)
def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@functools.lru_cache(None)
def base_bert():
    from transformers import BertModel, BertTokenizer
    ensure_files(BASE_MODEL)
    return BertTokenizer.from_pretrained(BASE_MODEL), BertModel.from_pretrained(BASE_MODEL).to(device()).eval()


@functools.lru_cache(None)
def load_species(sp):
    import joblib, torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    d = os.path.join(HERE, "models", SPECIES[sp])
    ensure_files(d, os.path.join(HERE, "token_pca.joblib"))
    ml = joblib.load(os.path.join(d, "ml_models_noRF.joblib"))
    pst = F.PSTNPss(); pst.z = ml["pstnp"]
    tok = AutoTokenizer.from_pretrained(os.path.join(d, "dnabert_model"))
    ft = BertForSequenceClassification.from_pretrained(os.path.join(d, "dnabert_model")).to(device()).eval()
    bil = BiLSTM(); bil.load_state_dict(torch.load(os.path.join(d, "bilstm.pt"), map_location="cpu")); bil.to(device()).eval()
    w = json.load(open(os.path.join(d, "ensemble_weights.json")))
    return dict(ml=ml, pst=pst, tok=tok, ft=ft, bilstm=bil, weights=w)


def BiLSTM(nt_dim=8, bert_dim=64, hidden=64, drop=0.3):
    """DNABERT-BiLSTM architecture (same as training): BiLSTM over nucleotide channels + BiLSTM over DNABERT tokens."""
    import torch, torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm_nt = nn.LSTM(nt_dim, hidden, batch_first=True, bidirectional=True)
            self.lstm_bert = nn.LSTM(bert_dim, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(8 * hidden, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))

        def forward(self, nt, bert):
            a, _ = self.lstm_nt(nt); b, _ = self.lstm_bert(bert)
            return self.head(torch.cat([a.mean(1), a.amax(1), b.mean(1), b.amax(1)], 1)).squeeze(1)
    return Net()


def score(seqs, sp, bs=512):
    """Per-model and ensemble 5mC probabilities for a list of 41-bp windows."""
    import joblib, torch
    M = load_species(sp); tok, bert = base_bert(); pca = joblib.load(os.path.join(HERE, "token_pca.joblib"))
    dev = device(); mean, tokp, p_ft = [], [], []
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            b = seqs[i:i + bs]
            h = bert(**tok([kmers(s) for s in b], return_tensors="pt", padding=True).to(dev)).last_hidden_state[:, 1:-1, :].float().cpu().numpy()
            mean.append(h.mean(1)); tokp.append(pca.transform(h.reshape(-1, 768)).reshape(len(b), 36, -1))
            e = M["tok"]([kmers(s) for s in b], padding="max_length", max_length=38, truncation=True, return_tensors="pt").to(dev)
            p_ft.append(torch.softmax(M["ft"](**e).logits.float(), -1)[:, 1].cpu().numpy())
    # training used float16-stored embeddings; round the same way
    mean = np.concatenate(mean).astype(np.float16).astype(np.float32)
    tokp = np.concatenate(tokp).astype(np.float16).astype(np.float32)
    ml = M["ml"]
    X = pd.concat([F.handcrafted(seqs, M["pst"]), pd.DataFrame(mean, columns=[f"BERT_{i}" for i in range(768)])], axis=1)[ml["all_features"]]
    Z = pd.DataFrame(ml["scaler"].transform(X.values), columns=ml["all_features"])[ml["selected"]].values
    out = pd.DataFrame({m: ml["models"][m].predict_proba(Z)[:, 1] for m in ("XGBoost", "LightGBM")})
    out["DNABERT"] = np.concatenate(p_ft)
    nt = F.per_position(seqs); pb = []
    with torch.no_grad():
        for i in range(0, len(seqs), 2048):
            pb.append(torch.sigmoid(M["bilstm"](torch.tensor(nt[i:i + 2048], device=dev), torch.tensor(tokp[i:i + 2048], device=dev))).cpu().numpy())
    out["DNABERT-BiLSTM"] = np.concatenate(pb)
    out["prob_5mC"] = sum(M["weights"][m] * out[m] for m in MODELS)
    return out


def predict(df, sp, mode="auto", threshold=0.5, chunk=20000):
    W, mode = windows(df, mode)
    if W.empty: return W.assign(prob_5mC=[], prediction=[]), mode
    parts = [score(W.window.iloc[i:i + chunk].tolist(), sp) for i in range(0, len(W), chunk)]
    P = pd.concat(parts, ignore_index=True)
    W = pd.concat([W, P.astype(np.float64).round(4)], axis=1)
    W["prediction"] = np.where(W.prob_5mC >= threshold, "5mC", "non-5mC")
    return W, mode


# ---------------------------------------------------------------- CLI (web interface: app.py)
def main():
    ap = argparse.ArgumentParser(description="WiMeP: 5mC site prediction for Oryza rufipogon and Aegilops tauschii")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="score sequences from a FASTA/CSV file")
    p.add_argument("--species", required=True, choices=sorted(SPECIES))
    p.add_argument("--input", required=True, help="FASTA, or CSV with a `sequence` column")
    p.add_argument("--out", required=True, help="output CSV")
    p.add_argument("--mode", default="auto", choices=["auto", "site", "scan"])
    p.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    if not os.path.exists(a.input): sys.exit(f"input not found: {a.input}")
    df = read_input(a.input); short = (df.sequence.str.len() < 41).sum()
    if short: print(f"warning: {short} sequence(s) shorter than 41 bp skipped", file=sys.stderr)
    res, mode = predict(df[df.sequence.str.len() >= 41], a.species, a.mode, a.threshold)
    res.to_csv(a.out, index=False)
    print(f"{mode} mode: {len(res)} cytosines scored, {(res.prediction == '5mC').sum()} predicted 5mC -> {a.out}")


if __name__ == "__main__":
    main()
