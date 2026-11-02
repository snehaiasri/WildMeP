#!/usr/bin/env python3
"""Write models/<species>/ml_models_noRF.joblib from the original ml_models.joblib (github.com/diprosinha/WiMeP).

The Random Forest takes over 1 GB of memory but has an ensemble weight of about 1e-9 or less, so the hosted app omits it.
usage: python slim_models.py <folder holding the original models/ directory>
"""
import os, sys
import torch  # noqa: F401  (first: provides libgomp for LightGBM where the system has none)
import joblib

src = sys.argv[1]
for sp in ("Oryza_rufipogon", "Aegilops_tauschii"):
    ml = joblib.load(os.path.join(src, "models", sp, "ml_models.joblib"))
    del ml["models"]["RF"]
    joblib.dump(ml, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", sp, "ml_models_noRF.joblib"), compress=3)
