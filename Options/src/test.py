import pandas as pd
import numpy as np
from pathlib import Path

f = pd.read_parquet("Options/data/processed/surface_factors.parquet")
f["quote_date"] = pd.to_datetime(f["quote_date"])
f = f.sort_values("quote_date")

col = "atm_90d_closest"
dx = f[col].diff()
y = dx.shift(-1)
print(y.describe())
print("fraction zero-ish:", np.mean(np.abs(y.dropna()) < 1e-8))
