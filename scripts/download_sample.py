#!/usr/bin/env python3
"""Write a small synthetic sample. This does not download upstream corpora."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minimol.data.synthetic import write_admet_sample, write_cds_sample, write_reg_sample

root = Path("data/processed")
print(write_reg_sample(root / "reg_sample"))
print(write_cds_sample(root / "cds_sample"))
print(write_admet_sample(root / "admet_sample"))
