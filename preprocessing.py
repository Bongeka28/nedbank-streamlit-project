"""Importable preprocessing functions for portable saved pipelines."""
import numpy as np
def signed_log(x):return np.sign(x)*np.log1p(np.abs(x))
