"""Handcrafted sequence encodings for 41-bp windows (feature groups of the WiMeP pipeline).

NCP    nucleotide chemical properties (ring structure, functional group, hydrogen bonding), 3 x 41
GC     GC content, 1
BIN    mono-nucleotide binary (one-hot) encoding, 4 x 41
ONF    oligonucleotide frequency, k = 1..4 (4 + 16 + 64 + 256)
PSTNP  position-specific trinucleotide propensity, single strand (PSTNPss), 39; fitted on training data only
EIIP   electron-ion interaction potential per position, 41
"""
import itertools
import numpy as np
import pandas as pd

NT = "ACGT"
IDX = {c: k for k, c in enumerate(NT)}
NCP = {"A": (1, 1, 1), "C": (0, 1, 0), "G": (1, 0, 0), "T": (0, 0, 1)}
EIIP = {"A": 0.1260, "C": 0.1340, "G": 0.0806, "T": 0.1335}
GROUPS = ["NCP", "GC", "BIN", "ONF", "PSTNP", "EIIP", "BERT"]


def to_index(seqs):
    return np.array([[IDX[c] for c in s] for s in seqs], dtype=np.int8)


def per_position(seqs):
    """41 x 8 per-nucleotide channels (one-hot 4, NCP 3, EIIP 1), used by the BiLSTM."""
    X = to_index(seqs)
    onehot = np.eye(4, dtype=np.float32)[X]
    ncp = np.array([NCP[c] for c in NT], np.float32)[X]
    eiip = np.array([EIIP[c] for c in NT], np.float32)[X][..., None] * 5  # rescaled to ~0.4-0.7
    return np.concatenate([onehot, ncp, eiip], axis=2)


class PSTNPss:
    def fit(self, seqs, y):
        X = to_index(seqs); L = X.shape[1] - 2
        tri = X[:, :-2] * 16 + X[:, 1:-1] * 4 + X[:, 2:]
        y = np.asarray(y)
        self.z = np.zeros((64, L))
        for lab, sign in ((1, 1.0), (0, -1.0)):
            t = tri[y == lab]
            for j in range(L):
                self.z[:, j] += sign * np.bincount(t[:, j], minlength=64) / len(t)
        return self

    def transform(self, seqs):
        X = to_index(seqs)
        tri = X[:, :-2] * 16 + X[:, 1:-1] * 4 + X[:, 2:]
        return self.z[tri, np.arange(tri.shape[1])]


def onf(seqs, kmax=4):
    cols, out = [], []
    for k in range(1, kmax + 1):
        kmers = ["".join(p) for p in itertools.product(NT, repeat=k)]
        pos = {m: i for i, m in enumerate(kmers)}
        M = np.zeros((len(seqs), len(kmers)), np.float32)
        for r, s in enumerate(seqs):
            for i in range(len(s) - k + 1):
                M[r, pos[s[i:i + k]]] += 1
        out.append(M / (len(seqs[0]) - k + 1)); cols += [f"ONF_{m}" for m in kmers]
    return np.hstack(out), cols


def handcrafted(seqs, pstnp):
    X = to_index(seqs); L = X.shape[1]
    ncp = np.array([NCP[c] for c in NT], np.float32)[X].reshape(len(seqs), -1)
    gc = np.isin(X, [1, 2]).mean(1, keepdims=True).astype(np.float32)
    binary = np.eye(4, dtype=np.float32)[X].reshape(len(seqs), -1)
    o, ocols = onf(seqs)
    ps = pstnp.transform(seqs).astype(np.float32)
    eiip = np.array([EIIP[c] for c in NT], np.float32)[X]
    cols = ([f"NCP_p{i + 1}_{n}" for i in range(L) for n in ("ring", "func", "hbond")] + ["GC_content"]
            + [f"BIN_p{i + 1}_{c}" for i in range(L) for c in NT] + ocols
            + [f"PSTNP_p{i + 1}" for i in range(L - 2)] + [f"EIIP_p{i + 1}" for i in range(L)])
    return pd.DataFrame(np.hstack([ncp, gc, binary, o, ps, eiip]), columns=cols)


def group_of(col):
    return col.split("_")[0] if not col.startswith("GC") else "GC"
