from dotenv import load_dotenv
load_dotenv()

import warnings
from pathlib import Path
import os
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import distributions, kstest
from matplotlib.ticker import MultipleLocator, MaxNLocator


def inverse_transform(description):
    if description.startswith("Normal"):
        mean = float(description.split("mean=")[1].split(",")[0])
        std_dev = float(description.split("stdDev=")[1].split(")")[0])
        return 'norm', (mean, std_dev)

    elif description.startswith("Exponential"):
        mean = float(description.split("mean=")[1].split(")")[0])
        loc = 0  # Exponential typically assumes loc=0
        scale = mean  # scale is equivalent to mean for Exponential
        return 'expon', (loc, scale)

    elif description.startswith("Gamma"):
        alpha = float(description.split("alpha=")[1].split(",")[0])
        beta = float(description.split("beta=")[1].split(")")[0])
        loc = 0  # Assuming loc=0 by default
        scale = beta
        return 'gamma', (alpha, loc, scale)

    elif description.startswith("Weibull"):
        shape = float(description.split("shape=")[1].split(",")[0])
        scale = float(description.split("scale=")[1].split(")")[0])
        loc = 0  # Assuming loc=0 by default
        return 'weibull_min', (shape, loc, scale)

    elif description.startswith("Lognormal"):
        normal_mean = float(description.split("normalMean=")[1].split(",")[0])
        normal_std_dev = float(description.split("normalStdDev=")[1].split(")")[0])
        s = normal_std_dev
        loc = 0  # Typically loc=0 for lognorm
        scale = np.exp(normal_mean)  # scale is exp(mean) for lognorm
        return 'lognorm', (s, loc, scale)

    elif description.startswith("Triangular"):
        min_val = float(description.split("min=")[1].split(",")[0])
        mode_val = float(description.split("mode=")[1].split(",")[0])
        max_val = float(description.split("max=")[1].split(")")[0])
        loc = min_val
        scale = max_val - min_val
        c = (mode_val - min_val) / scale
        return 'triang', (c, loc, scale)

    else:
        raise ValueError(f"Unsupported description: {description}")
    

def get_dist_name(description):
    if description.startswith("Normal"):
        return 'norm'
    elif description.startswith("Exponential"):
        return 'expon'
    elif description.startswith("Gamma"):
        return 'gamma'
    elif description.startswith("Weibull"):
        return 'weibull_min'
    elif description.startswith("Lognormal"):
        return 'lognorm'
    elif description.startswith("Triangular"):
        return 'triang'
    else:
        raise ValueError(f"Unsupported description: {description}")
    

def get_representitives(data: pd.Series):
    return {
        "mean": data.mean(),
        "median": data.median(),
        "mode": data.mode().values[0],
    }

def vote_representitives(prefix1: str, prefix2: str, representitive1: dict, representitive2: dict):
    result = {prefix1: [], prefix2: []} # weekday, weekend
    if representitive1["mean"] > representitive2["mean"]:
        result[prefix1].append("mean")
    else:
        result[prefix2].append("mean") # 같기만해도 주말이 더 높다고 판단
    
    if representitive1["median"] > representitive2["median"]:
        result[prefix1].append("median")
    else:
        result[prefix2].append("median")
    
    if representitive1["mode"] > representitive2["mode"]:
        result[prefix1].append("mode")
    else:
        result[prefix2].append("mode")
    
    won = prefix1 if len(result[prefix1]) > len(result[prefix2]) else prefix2
    return result, won


def transform_params(dist_name, params):
    if dist_name == 'norm':
        loc, scale = params
        return f"Normal(mean={loc:.3f}, stdDev={scale:.3f})"
    # elif dist_name == 'poisson': # Poisson distribution is not supported for fitting
    #     loc, scale = params  # Poisson uses loc=mean, scale is ignored
    #     return f"Poisson(mean={loc:.3f})"
    elif dist_name == 'expon':
        loc, scale = params
        return f"Exponential(mean={scale:.3f})"
    elif dist_name == 'gamma':
        a, loc, scale = params
        beta = scale
        alpha = a
        return f"Gamma(alpha={alpha:.3f}, beta={beta:.3f})"
    # elif dist_name == 'beta': # Beta distribution is for data between 0 and 1, not suitable for this data
    #     a, b, loc, scale = params
    #     return f"Beta(alpha1={a:.3f}, alpha2={b:.3f})"
    elif dist_name == 'weibull_min': # RuntimeWarning: overflow encountered in divide return np.sum((1 + np.log(shifted/scale)/shape**2)/shifted)
        c, loc, scale = params
        return f"Weibull(shape={c:.3f}, scale={scale:.3f})"
    elif dist_name == 'lognorm':
        s, loc, scale = params
        mu = np.log(scale)
        sigma = s
        return f"Lognormal(normalMean={mu:.3f}, normalStdDev={sigma:.3f})"
    elif dist_name == 'triang':
        c, loc, scale = params
        min_val = loc
        max_val = loc + scale
        mode_val = loc + c * scale
        return f"Triangular(min={min_val:.3f}, mode={mode_val:.3f}, max={max_val:.3f})"
    else:
        raise ValueError(f"Unsupported distribution: {dist_name}")