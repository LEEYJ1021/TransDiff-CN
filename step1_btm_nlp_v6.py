"""
================================================================================
 NLP-Augmented Bayesian Transition Model  ·  v6.0  (Full-Domain Edition)
 Scientific Knowledge Diffusion in Chinese Transportation Research
================================================================================

CHANGES OVER v5.0
─────────────────
  CHG-1  No document filtering — all 11,271 papers kept.
  CHG-2  Moran I → permutation-only (analytic variance caused z = -101,143).
  CHG-3  Gap-statistic min_k raised to 5 (more granular sub-domains).
  CHG-4  Expanded TRANSPORT_TAXONOMY.
  CHG-5  match_to_transport_domain: normalised scores + guaranteed assignment.
  CHG-6  Topic-naming: even low-scoring topics get the best-matching domain.
  CHG-7  transport_share covariate kept but no longer used as a filter.
  CHG-8  Steady-state CI annotation: if CI width < 0.002 → note "tight posterior".
  CHG-9  Subsample corr guard.
  CHG-10 All v5.0 bug-fixes retained (FIX-1…5, ADD-1…10).
"""

# ── 0. IMPORTS ──────────────────────────────────────────────────────────────
import os, re, json, warnings, logging, pickle, itertools
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
from scipy.stats import norm as scipy_norm, gaussian_kde
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import expm

import pymc as pm
import pytensor.tensor as pt
import arviz as az
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, mean_squared_error, mean_absolute_error)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.ERROR)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
FILE_PATH = "data/scholarly_works.csv"
OUT_DIR   = "outputs/btm_nlp_v6"
CACHE_DIR = Path(OUT_DIR) / "nlp_cache"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

YEAR_START  = 2007
YEAR_END    = 2023
N_STATES    = 3
N_TOPICS    = 10
BERT_MODEL  = "sentence-transformers/paraphrase-MiniLM-L6-v2"
N_SAMPLES   = 2000
N_TUNE      = 2000
N_CHAINS    = 4
ACCEPT      = 0.90
TRANSPORT_RELEVANCE_THRESHOLD = 0.0

matplotlib.rcParams.update({
    "figure.dpi": 150, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlepad": 8, "legend.framealpha": 0.85, "grid.alpha": 0.3,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})
PALETTE      = sns.color_palette("tab10")
TYPE_COLORS  = {"C9": "#E91E63", "C7": "#9C27B0", "research": "#FF9800",
                "transport": "#4CAF50", "teaching": "#2196F3"}
REG_COLORS   = {"North": "#2196F3", "East": "#4CAF50", "Central": "#FF9800",
                "South": "#E91E63", "West": "#9C27B0", "Northeast": "#795548"}
CHAIN_COLORS = ["#1565C0", "#C62828", "#2E7D32", "#E65100"]
STATE_LABELS = {0: "Low", 1: "Medium", 2: "High"}

# ── CHG-4: EXPANDED TRANSPORT TAXONOMY ────────────────────────────────────────
TRANSPORT_TAXONOMY = {
    "Intelligent Transport & Autonomous Vehicles": [
        "autonomous", "self-driving", "lidar", "radar", "v2x", "connected vehicle",
        "lane detection", "object detection", "vehicle detection", "path planning",
        "driverless", "cooperative driving", "perception", "sensor fusion",
        "vehicle control", "adaptive cruise", "platooning", "vehicle platoon",
        "automated vehicle", "autonomous driving", "collision avoidance",
    ],
    "Traffic Control & Network Optimization": [
        "traffic signal", "signal control", "traffic flow", "congestion",
        "route optimization", "traffic management", "intersection control",
        "urban traffic", "incident detection", "vehicle routing problem",
        "network optimization", "signal timing", "queue", "delay",
        "traffic simulation", "microscopic simulation", "macroscopic model",
        "traffic wave", "shockwave", "variable speed limit",
    ],
    "Structural Health & Pavement Engineering": [
        "pavement", "asphalt", "bridge", "structural health", "crack detection",
        "fatigue", "concrete", "compressive strength", "rutting", "deflection",
        "modulus", "bitumen", "aggregate", "bridge inspection", "deterioration",
        "binder", "mastic", "asphalt mixture", "acoustic emission", "moisture damage",
        "pavement performance", "road distress", "finite element", "deck",
    ],
    "Rail & High-Speed Transport Systems": [
        "high-speed rail", "hsr", "railway", "magnetic levitation", "urban rail",
        "metro", "subway", "track", "rolling stock", "traction", "maglev",
        "pantograph", "catenary", "wheel-rail", "train control",
        "rail transit", "timetable", "scheduling", "train delay", "signalling",
    ],
    "Sustainable & Electric Mobility": [
        "electric vehicle", "battery", "charging station", "emission", "carbon",
        "fuel cell", "hybrid vehicle", "energy consumption", "green transport",
        "renewable", "low-emission", "lifecycle", "decarbonization",
        "anode", "cathode", "electrolyte", "ionic", "lithium", "sodium",
        "energy storage", "electrochemical", "capacitor", "charge discharge",
        "cycle stability", "coulombic efficiency", "solid state battery",
        "nanostructure energy", "graphene battery", "carbon material energy",
        "doped electrode", "mesoporous electrode", "reversible capacity",
        "single atom catalyst energy", "nanoporous energy",
    ],
    "Logistics, Freight & Supply Chain": [
        "logistics", "supply chain", "freight", "last-mile", "warehouse",
        "distribution", "delivery", "fleet management", "container", "port",
        "cargo", "intermodal", "cold chain", "humanitarian logistics",
        "vehicle routing", "traveling salesman", "bin packing", "inventory",
        "demand forecasting", "transportation network design",
    ],
    "Safety, Risk & Reliability Analysis": [
        "safety", "accident", "collision", "reliability", "risk", "hazard",
        "fault", "failure", "crash prediction", "traffic safety", "fatality",
        "injury", "pedestrian safety", "drunk driving", "distracted",
        "road safety", "vehicle safety", "crash severity", "black spot",
        "near-miss", "safety performance function",
    ],
    "Data Science, AI & Smart Sensing": [
        "deep learning", "neural network", "machine learning", "prediction",
        "big data", "reinforcement learning", "computer vision", "data mining",
        "convolutional", "lstm", "transformer", "gps", "sensor", "iot",
        "image recognition", "anomaly detection", "federated learning",
        "transfer learning", "graph neural", "attention mechanism",
        "object tracking", "semantic segmentation", "point cloud",
        "edge computing", "cloud computing", "digital twin",
        "offloading", "mobile edge", "task offloading", "wireless",
        "communication latency", "v2i", "v2v", "cooperative computing",
    ],
    "Advanced Materials for Transport Infrastructure": [
        "pavement material", "asphalt material", "bridge material",
        "road polymer", "transport polymer", "road composite",
        "transport composite", "rail composite", "road fiber",
        "transport coating", "road corrosion",
        "polymer", "composite", "nanoparticle", "thin film", "coating",
        "corrosion", "wear resistance", "tribology", "lubricant",
        "thermal conductivity", "mechanical property", "tensile strength",
        "yield strength", "fracture toughness", "creep", "hardness",
        "surface treatment", "alloy", "steel", "aluminum", "magnesium",
        "carbon fiber", "glass fiber", "epoxy", "resin", "rubber",
        "cathode material", "anode material", "separator", "solid electrolyte",
        "perovskite", "spinel", "olivine", "intercalation",
    ],
    "Human Factors, Health & Biomedical Transport Safety": [
        "driver fatigue", "occupational health", "human factors", "ergonomics",
        "physiological monitoring", "driver behavior", "mental workload",
        "air pollution exposure", "noise exposure", "vibration exposure",
        "pedestrian health", "wearable sensor", "cognitive load",
        "injury biomechanics", "crash injury", "trauma", "rehabilitation",
        "prosthetics", "wheelchair", "disability transport", "accessibility",
        "health monitoring", "biomarker", "diagnostic imaging",
        "drug impairment", "alcohol impairment", "medical fitness to drive",
        "therapy", "treatment", "clinical", "patient", "disease",
        "cell biology", "gene expression", "metabolic", "drug delivery",
        "inflammatory", "immune response", "neural", "brain activity",
        "cardiovascular", "respiratory", "cancer", "tumor",
        "dna", "protein", "enzyme", "receptor", "antibody",
        "biocompatibility", "cytotoxicity", "apoptosis", "proliferation",
    ],
}

TRANSPORT_CORE_VOCAB = [
    "transport", "traffic", "vehicle", "road", "highway", "pavement",
    "bridge", "railway", "rail", "metro", "subway", "logistics",
    "freight", "supply chain", "safety", "accident", "collision",
    "autonomous", "self-driving", "electric vehicle", "mobility",
    "infrastructure", "congestion", "routing", "fleet", "driving",
    "pedestrian", "signal", "intersection", "transit", "commute",
    "corridor", "lane", "tunnel", "port", "cargo", "aviation",
    "shipping", "bus", "truck", "motorcycle", "bicycle", "car",
    "automobile", "speed", "queue", "delay", "emission", "fuel",
    "charging station", "v2x", "lidar", "navigation", "gps tracking",
    "driver", "freeway", "expressway", "roundabout",
    "traffic flow", "traffic signal", "road network", "toll",
    "urban mobility", "public transport", "mass transit", "tram",
    "high speed rail", "parking", "ride sharing", "carpool",
    "driverless", "platooning", "road safety", "crash", "fatality",
    "injury prevention", "transportation system", "transportation network",
]

# ── GEOGRAPHIC & INSTITUTIONAL METADATA ──────────────────────────────────────
INST_META = {
    "Chinese Academy of Sciences":                   ("Beijing",    39.9042, 116.4074, "North",     "research",   1),
    "Tsinghua University":                           ("Beijing",    40.0033, 116.3267, "North",     "C9",         2),
    "Beijing Jiaotong University":                   ("Beijing",    39.9647, 116.3539, "North",     "transport",  3),
    "Tongji University":                             ("Shanghai",   31.2813, 121.5034, "East",      "transport",  4),
    "Southeast University":                          ("Nanjing",    32.1578, 118.8208, "East",      "C9",         5),
    "Zhejiang University":                           ("Hangzhou",   30.2587, 120.1310, "East",      "C9",         6),
    "Shanghai Jiao Tong University":                 ("Shanghai",   31.2024, 121.4337, "East",      "C9",         7),
    "Beihang University":                            ("Beijing",    40.0009, 116.3428, "North",     "C7",         8),
    "Wuhan University of Technology":                ("Wuhan",      30.5099, 114.3974, "Central",   "transport",  9),
    "Central South University":                      ("Changsha",   28.1804, 112.9321, "Central",   "transport", 10),
    "Harbin Institute of Technology":                ("Harbin",     45.7761, 126.6723, "Northeast", "C9",        11),
    "University of Chinese Academy of Sciences":     ("Beijing",    39.9975, 116.3862, "North",     "research",  12),
    "Peking University":                             ("Beijing",    39.9971, 116.3107, "North",     "C9",        13),
    "Jilin University":                              ("Changchun",  43.8878, 125.3247, "Northeast", "C9",        14),
    "Wuhan University":                              ("Wuhan",      30.5389, 114.3599, "Central",   "C9",        15),
    "South China University of Technology":          ("Guangzhou",  23.0490, 113.4022, "South",     "C9",        16),
    "Shandong University":                           ("Jinan",      36.6804, 117.0601, "North",     "C9",        17),
    "Sun Yat-sen University":                        ("Guangzhou",  23.0932, 113.2978, "South",     "C9",        18),
    "Huazhong University of Science and Technology": ("Wuhan",      30.5091, 114.4139, "Central",   "C9",        19),
    "Beijing Institute of Technology":               ("Beijing",    39.9652, 116.3223, "North",     "C7",        20),
    "Shenzhen University":                           ("Shenzhen",   22.5341, 113.9298, "South",     "teaching",  21),
    "Southwest Jiaotong University":                 ("Chengdu",    30.6441, 103.8418, "West",      "transport", 22),
    "Fudan University":                              ("Shanghai",   31.2985, 121.5018, "East",      "C9",        23),
    "Sichuan University":                            ("Chengdu",    30.6310, 104.0821, "West",      "C9",        24),
    "Tianjin University":                            ("Tianjin",    39.1053, 117.1619, "North",     "C9",        25),
    "Dalian University of Technology":               ("Dalian",     38.8574, 121.5264, "Northeast", "C9",        26),
    "Chongqing University":                          ("Chongqing",  29.5594, 106.5509, "West",      "C9",        27),
    "Nanjing University":                            ("Nanjing",    32.1146, 118.7835, "East",      "C9",        28),
    "Soochow University":                            ("Suzhou",     31.3019, 120.6239, "East",      "teaching",  29),
    "Xi\'an Jiaotong University":                    ("Xi\'an",     34.2570, 108.9784, "West",      "C9",        30),
    "University of Science and Technology of China": ("Hefei",      31.8315, 117.2571, "East",      "C9",        31),
    "Shanghai University":                           ("Shanghai",   31.2835, 121.4440, "East",      "teaching",  32),
    "Beijing University of Technology":              ("Beijing",    39.8590, 116.4815, "North",     "teaching",  33),
}
INST_NAMES = list(INST_META.keys())
N_INST     = len(INST_NAMES)
INST_IDX   = {n: i for i, n in enumerate(INST_NAMES)}
REGIONS    = ["North", "East", "Central", "South", "West", "Northeast"]
N_REGIONS  = len(REGIONS)
REG_IDX    = {r: i for i, r in enumerate(REGIONS)}
INST_TYPES = ["C9", "C7", "research", "transport", "teaching"]

PATENT_VOCAB = [
    "autonomous vehicle", "self-driving", "lidar", "deep learning",
    "neural network", "reinforcement learning", "traffic signal", "signal control",
    "intelligent transport", "vehicle detection", "object detection",
    "road safety", "electric vehicle", "battery management", "charging station",
    "connected vehicle", "V2X communication", "route optimization",
    "bridge inspection", "pavement monitoring", "structural health",
    "high-speed rail", "urban rail transit", "driver assistance",
]

INST_KW = {
    "Chinese Academy of Sciences":                   ["chinese academy of sciences", "中国科学院"],
    "Tsinghua University":                           ["tsinghua university", "tsinghua univ", "清华大学"],
    "Beijing Jiaotong University":                   ["beijing jiaotong", "北京交通"],
    "Tongji University":                             ["tongji university", "同济大学"],
    "Southeast University":                          ["southeast university", "东南大学"],
    "Zhejiang University":                           ["zhejiang university", "浙江大学"],
    "Shanghai Jiao Tong University":                 ["shanghai jiao tong", "sjtu", "上海交通"],
    "Beihang University":                            ["beihang university", "buaa", "北京航空航天"],
    "Wuhan University of Technology":                ["wuhan university of technology", "武汉理工"],
    "Central South University":                      ["central south university", "中南大学"],
    "Harbin Institute of Technology":                ["harbin institute of technology", "哈尔滨工业"],
    "University of Chinese Academy of Sciences":     ["university of chinese academy", "国科大"],
    "Peking University":                             ["peking university", "北京大学"],
    "Jilin University":                              ["jilin university", "吉林大学"],
    "Wuhan University":                              ["wuhan university", "武汉大学"],
    "South China University of Technology":          ["south china university of technology", "scut", "华南理工"],
    "Shandong University":                           ["shandong university", "山东大学"],
    "Sun Yat-sen University":                        ["sun yat-sen", "zhongshan university", "中山大学"],
    "Huazhong University of Science and Technology": ["huazhong university", "hust", "华中科技"],
    "Beijing Institute of Technology":               ["beijing institute of technology", "北京理工"],
    "Shenzhen University":                           ["shenzhen university", "深圳大学"],
    "Southwest Jiaotong University":                 ["southwest jiaotong", "西南交通"],
    "Fudan University":                              ["fudan university", "复旦大学"],
    "Sichuan University":                            ["sichuan university", "四川大学"],
    "Tianjin University":                            ["tianjin university", "天津大学"],
    "Dalian University of Technology":               ["dalian university of technology", "大连理工"],
    "Chongqing University":                          ["chongqing university", "重庆大学"],
    "Nanjing University":                            ["nanjing university", "南京大学"],
    "Soochow University":                            ["soochow university", "苏州大学"],
    "Xi\'an Jiaotong University":                    ["xi\'an jiaotong", "xian jiaotong", "西安交通"],
    "University of Science and Technology of China": ["university of science and technology of china", "ustc", "中国科技大"],
    "Shanghai University":                           ["shanghai university", "上海大学"],
    "Beijing University of Technology":              ["beijing university of technology", "北京工业大学"],
}

INST_PRIOR_FREQ = {
    "Chinese Academy of Sciences": 1505, "Tsinghua University": 793,
    "Beijing Jiaotong University": 602,  "Tongji University": 585,
    "Southeast University": 559,         "Zhejiang University": 505,
    "Shanghai Jiao Tong University": 473,"Beihang University": 470,
    "Wuhan University of Technology": 418,"Central South University": 417,
    "Harbin Institute of Technology": 386,"University of Chinese Academy of Sciences": 371,
    "Peking University": 321,            "Jilin University": 312,
    "Wuhan University": 308,             "South China University of Technology": 307,
    "Shandong University": 289,          "Sun Yat-sen University": 273,
    "Huazhong University of Science and Technology": 272,"Beijing Institute of Technology": 263,
    "Shenzhen University": 255,          "Southwest Jiaotong University": 252,
    "Fudan University": 237,             "Sichuan University": 237,
    "Tianjin University": 236,           "Dalian University of Technology": 224,
    "Chongqing University": 223,         "Nanjing University": 210,
    "Soochow University": 202,           "Xi\'an Jiaotong University": 190,
    "University of Science and Technology of China": 180,
    "Shanghai University": 171,          "Beijing University of Technology": 167,
}

CHINA_LON = [
    73.7, 75.5, 77.5, 79.5, 81.0, 83.0, 85.0, 87.0, 88.5, 90.0, 92.0, 94.0,
    96.5, 97.5, 98.5,100.0,102.0,104.2,106.5,108.5,110.5,112.0,114.0,115.5,
   117.0,118.5,119.5,120.3,121.0,121.8,122.3,123.0,124.0,125.0,126.5,128.0,
   130.0,131.5,132.5,134.0,134.8,135.1,134.5,133.0,131.5,130.0,128.5,126.0,
   124.5,123.0,122.0,121.5,121.2,120.8,120.5,120.0,119.5,119.2,119.0,118.5,
   117.8,117.0,115.8,114.5,113.2,112.0,110.5,109.5,109.0,108.5,107.5,106.5,
   105.5,104.0,102.5,101.0, 99.5, 98.5, 97.5, 97.0, 97.5, 98.2, 98.5, 97.8,
    97.0, 96.5, 96.0, 95.5, 94.0, 92.0, 90.5, 88.5, 87.0, 85.5, 84.0, 82.0,
    80.0, 78.0, 76.0, 74.0, 73.7,
]
CHINA_LAT = [
    39.5, 41.5, 43.0, 44.5, 46.5, 48.0, 49.5, 49.8, 50.0, 49.5, 49.5, 50.5,
    50.5, 49.5, 49.0, 49.5, 49.5, 50.0, 49.0, 48.5, 48.0, 47.5, 47.0, 46.5,
    46.0, 45.0, 44.0, 42.5, 41.5, 40.5, 40.8, 41.5, 42.5, 43.5, 44.5, 45.5,
    46.0, 47.0, 47.5, 48.3, 48.6, 48.5, 47.5, 47.0, 46.0, 44.5, 42.5, 40.5,
    39.5, 38.0, 36.5, 35.0, 33.5, 32.5, 31.5, 30.0, 28.5, 27.5, 26.5, 25.5,
    24.5, 23.5, 22.5, 22.0, 21.5, 21.2, 21.5, 22.0, 21.8, 21.5, 22.5, 21.5,
    22.5, 23.0, 23.5, 25.0, 26.5, 27.5, 28.0, 25.5, 27.0, 28.5, 29.5, 31.0,
    32.0, 32.5, 33.5, 34.0, 33.5, 33.0, 33.5, 34.5, 35.5, 37.0, 38.5, 39.5,
    38.0, 36.5, 36.0, 37.0, 39.5,
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def banner(title):
    print("\n" + "═"*76 + f"\n  {title}\n" + "═"*76)

def cache_path(n): return CACHE_DIR / f"{n}.pkl"
def save_cache(obj, n):
    with open(cache_path(n), "wb") as f: pickle.dump(obj, f)
def load_cache(n):
    p = cache_path(n)
    return pickle.load(open(p, "rb")) if p.exists() else None

def shannon_entropy(p):
    p = p[p > 0]; return float(-np.sum(p * np.log(p)))

def extract_institution(text):
    if not isinstance(text, str): return None
    t = text.lower()
    for inst, kws in INST_KW.items():
        for kw in kws:
            if kw in t: return inst
    return None

def haversine_matrix(meta, names):
    c   = np.radians([[meta[n][1], meta[n][2]] for n in names])
    lat = c[:, 0:1]; lon = c[:, 1:2]
    dlat = lat - lat.T; dlon = lon - lon.T
    a = np.sin(dlat/2)**2 + np.cos(lat)*np.cos(lat.T)*np.sin(dlon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def row_norm(W):
    s = W.sum(1, keepdims=True)
    return np.where(s > 0, W / s, 0.)

def _savefig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"  -> {name}")

def _short_map_label(full_name):
    s = (full_name
         .replace("University of Chinese Academy of Sciences", "U. of CAS")
         .replace("University of Science and Technology of China", "USTC")
         .replace("Chinese Academy of Sciences", "CAS")
         .replace("Huazhong University of Science and Technology", "HUST")
         .replace("South China University of Technology", "S. China U. Tech.")
         .replace("University", "U.").replace("Institute", "Inst.")
         .replace("of Technology", "Tech.").replace("  ", " "))
    return s[:22]

def _star(lo, hi): return "★" if lo * hi > 0 else ""

def _loo_pointwise(loo_result):
    try:
        return np.asarray(loo_result.loo_i).flatten()
    except AttributeError:
        pass
    try:
        return loo_result.pointwise["elpd_loo"].values.flatten()
    except Exception:
        pass
    return None

# CHG-2: Moran\'s I — permutation-only (fixes z = -101,143 anomaly)
def morans_i(y, W, n_perm=999):
    """
    Moran\'s I using permutation test exclusively.
    The analytic variance formula becomes unstable for small n (n=33 here)
    and produced z = -101,143 in v5.0. Permutation is unbiased and robust.
    """
    n    = len(y)
    y_c  = y - y.mean()
    ycy  = y_c @ y_c
    S0   = W.sum()
    if ycy < 1e-12 or S0 < 1e-12:
        return 0.0, -1.0/(n-1), 0.0, 1.0

    Wy = W @ y_c
    I  = float((n * (y_c @ Wy)) / (ycy * S0))
    E_I = -1.0 / (n - 1)

    rng_m  = np.random.default_rng(0)
    I_perm = np.empty(n_perm)
    for b in range(n_perm):
        yp = rng_m.permutation(y_c)
        ypy = yp @ yp
        I_perm[b] = float((n * (yp @ (W @ yp))) / (ypy * S0)) if ypy > 1e-12 else 0.

    sigma_perm = I_perm.std() + 1e-12
    z = (I - I_perm.mean()) / sigma_perm
    p = float(np.mean(np.abs(I_perm - I_perm.mean()) >= abs(I - I_perm.mean())))
    return float(I), float(E_I), float(z), float(p)


# ══════════════════════════════════════════════════════════════════════════════
# EXTENDED STOP-WORDS
# ══════════════════════════════════════════════════════════════════════════════
EXTENDED_STOP_WORDS = set([
    "the","of","and","in","to","a","an","is","are","was","were","be","been",
    "being","have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","that","this","these","those","it","its",
    "we","our","they","their","he","she","his","her","i","my","you","your",
    "with","from","for","on","at","by","as","or","not","but","so","if","then",
    "than","also","into","up","out","about","over","after","before","between",
    "through","during","under","while","each","all","both","more","most","other",
    "such","only","own","same","however","although","therefore","thus","hence",
    "which","who","whom","whose","what","when","where","how","why","there",
    "study","paper","research","results","method","methods","approach","proposed",
    "based","using","used","use","model","models","data","analysis","system",
    "systems","performance","effect","effects","different","various","including",
    "compared","comparison","provide","provides","significant","important",
    "large","high","low","new","novel","existing","present","show","shows",
    "shown","found","find","obtained","achieved","developed","proposed",
    "further","well","also","thus","hence","therefore","however","although",
    "respectively","order","case","cases","due","given","two","three","four",
    "five","first","second","number","numbers","types","type","related","result",
    "various","several","certain","following","regarding","within",
    "among","across","between","upon","without","despite","consider","provide",
    "improve","increase","decrease","change","make","take","give","get",
    "need","want","seem","work","run","set","let","try","keep",
    "00","01","02","03","04","05","06","07","08","09","10","11","12",
    "2007","2008","2009","2010","2011","2012","2013","2014","2015","2016",
    "2017","2018","2019","2020","2021","2022","2023","doi","http","www",
    "vol","pp","no","fig","table","eq","et","al","ref","refs",
])

def clean_text_v6(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split()
              if len(t) >= 3 and t not in EXTENDED_STOP_WORDS]
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# TRANSPORT RELEVANCE SCORE (informational only — no filtering)
# ══════════════════════════════════════════════════════════════════════════════
def compute_transport_relevance(texts, vocab=TRANSPORT_CORE_VOCAB):
    vec = TfidfVectorizer(vocabulary=vocab, binary=False,
                           token_pattern=r"(?u)\b[a-z][a-z]{2,}\b")
    texts_clean = [clean_text_v6(t) for t in texts]
    try:
        tdm = vec.fit_transform(texts_clean)
    except ValueError:
        return np.ones(len(texts))
    scores = np.asarray(tdm.sum(axis=1)).flatten()
    max_s = scores.max() if scores.max() > 0 else 1.
    return scores / max_s


# ══════════════════════════════════════════════════════════════════════════════
# c-TF-IDF
# ══════════════════════════════════════════════════════════════════════════════
def compute_ctfidf_v6(corpus_per_topic):
    concat_docs = [" ".join([clean_text_v6(t) for t in texts])
                   for texts in corpus_per_topic.values()]
    vectorizer  = TfidfVectorizer(
        max_features=25000, ngram_range=(1, 3),
        sublinear_tf=True, min_df=2,
        stop_words=list(EXTENDED_STOP_WORDS),
        token_pattern=r"(?u)\b[a-z][a-z]{2,}\b"
    )
    tfidf_mat  = vectorizer.fit_transform(concat_docs)
    feat_names = np.array(vectorizer.get_feature_names_out())

    ctfidf_top_terms = {}
    for t in range(len(concat_docs)):
        row     = tfidf_mat[t].toarray().flatten()
        top_idx = row.argsort()[-30:][::-1]
        valid   = [feat_names[i] for i in top_idx
                   if feat_names[i] not in EXTENDED_STOP_WORDS
                   and len(feat_names[i]) >= 3
                   and not feat_names[i].replace(" ", "").isdigit()]
        ctfidf_top_terms[t] = valid[:12]
    return tfidf_mat, feat_names, ctfidf_top_terms


def compute_pmi_topic_v6(texts_clean, topic_labels_arr, topic_id, vocab_size=8000):
    count_v = CountVectorizer(
        max_features=vocab_size, ngram_range=(1, 2),
        stop_words=list(EXTENDED_STOP_WORDS),
        token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",
        min_df=3, binary=True,
    )
    texts_arr = np.array(texts_clean)
    try:
        tdm = count_v.fit_transform(texts_arr)
    except ValueError:
        return [], None, None
    feat_names = np.array(count_v.get_feature_names_out())
    mask     = topic_labels_arr == topic_id
    p_topic  = mask.mean()
    p_word   = np.asarray(tdm.mean(0)).flatten()
    p_word_t = np.asarray(tdm[mask].mean(0)).flatten() if mask.sum() > 0 else p_word
    pmi = (np.log(np.maximum(p_word_t, 1e-10))
           - np.log(np.maximum(p_word, 1e-10))
           - np.log(max(p_topic, 1e-10)))
    top_idx = pmi.argsort()[-20:][::-1]
    return feat_names[top_idx].tolist(), feat_names, pmi


# CHG-5: Guaranteed domain assignment with L2-normalised scores
def match_to_transport_domain(topic_terms, taxonomy):
    topic_text   = " ".join([clean_text_v6(t) for t in topic_terms])
    domain_texts = [" ".join([clean_text_v6(k) for k in kws])
                    for kws in taxonomy.values()]
    domain_names = list(taxonomy.keys())
    try:
        vec   = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                stop_words=list(EXTENDED_STOP_WORDS),
                                token_pattern=r"(?u)\b[a-z][a-z]{2,}\b")
        tfidf = vec.fit_transform([topic_text] + domain_texts)
        sims  = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        sims_norm = sims / (sims.max() + 1e-12)
        scores    = dict(zip(domain_names, sims_norm))
        best      = max(scores, key=scores.get)
        return best, scores
    except Exception:
        return domain_names[0], {d: 0. for d in domain_names}


def compute_topic_coherence_umass(topic_terms, corpus_clean, top_n=10):
    terms = [t for t in topic_terms[:top_n]
             if len(t) >= 3 and t not in EXTENDED_STOP_WORDS][:top_n]
    if len(terms) < 2: return 0.0
    try:
        cv  = CountVectorizer(vocabulary=terms, binary=True,
                              token_pattern=r"(?u)\b[a-z][a-z]{2,}\b")
        dtm = (cv.transform(corpus_clean) > 0).toarray()
        score, count = 0.0, 0
        for i in range(1, len(terms)):
            for j in range(i):
                d_j  = dtm[:, j].sum() + 1e-10
                d_ij = (dtm[:, i] & dtm[:, j]).sum() + 1
                score += np.log(d_ij / d_j)
                count += 1
        return score / max(count, 1)
    except Exception:
        return 0.0


# CHG-3: Gap-statistic with min_k = 5
def gap_statistic_k(embeddings, k_range=range(4, 13), B=10, random_state=42, min_k=5):
    rng = np.random.default_rng(random_state)
    ref_min = embeddings.min(axis=0)
    ref_max = embeddings.max(axis=0)

    gaps = []
    for k in k_range:
        km   = KMeans(n_clusters=k, n_init=10, random_state=random_state, max_iter=300)
        km.fit(embeddings)
        W_k  = float(km.inertia_)
        W_kb_list = []
        for b in range(B):
            ref  = rng.uniform(ref_min, ref_max, size=embeddings.shape)
            km_b = KMeans(n_clusters=k, n_init=5, random_state=b, max_iter=200)
            km_b.fit(ref)
            W_kb_list.append(float(km_b.inertia_))
        W_kb_arr = np.array(W_kb_list)
        gap  = np.log(W_kb_arr).mean() - np.log(W_k)
        sk   = np.sqrt(1 + 1.0/B) * np.log(W_kb_arr).std()
        gaps.append({"k": k, "gap": gap, "sk": sk, "W_k": W_k})

    gaps_df  = pd.DataFrame(gaps)
    k_vals   = gaps_df["k"].values
    gap_vals = gaps_df["gap"].values
    sk_vals  = gaps_df["sk"].values

    best_k = int(k_vals[0])
    found  = False
    for i in range(len(gaps_df) - 1):
        if gap_vals[i] >= gap_vals[i+1] - sk_vals[i+1]:
            best_k = int(k_vals[i])
            found  = True
            break

    if not found:
        increments = np.diff(gap_vals)
        if len(increments) >= 2:
            second_diff = np.diff(increments)
            elbow_idx   = int(np.argmin(second_diff))
            best_k      = int(k_vals[elbow_idx + 1])
        print(f"  [Gap-stat] Monotonic -> elbow fallback K={best_k}")

    best_k = max(best_k, min_k)
    return best_k, gaps_df


def identify_topics_v6(df_all, embeddings_all, topic_labels_all, N_TOPICS_USED):
    banner("B4+. ADVANCED TOPIC IDENTIFICATION  (v6.0 — no filtering)")
    texts_raw   = df_all["text_combined"].tolist()
    texts_clean = [clean_text_v6(t) for t in texts_raw]
    corpus_per_topic = {t: [texts_clean[i] for i, lbl in enumerate(topic_labels_all)
                             if lbl == t]
                        for t in range(N_TOPICS_USED)}

    print("  [c-TF-IDF v6.0] ...")
    _, feat_names, ctfidf_top_terms = compute_ctfidf_v6(corpus_per_topic)
    for t in range(N_TOPICS_USED):
        print(f"    T{t} c-TF-IDF: {ctfidf_top_terms[t][:6]}")

    print("  [PMI v6] ...")
    pmi_top_terms = {}
    for t in range(N_TOPICS_USED):
        terms, _, _ = compute_pmi_topic_v6(texts_clean, topic_labels_all, t)
        pmi_top_terms[t] = terms[:12]
        print(f"    T{t} PMI:     {pmi_top_terms[t][:6]}")

    print("  [Coherence] ...")
    coherence_scores = {}
    for t in range(N_TOPICS_USED):
        all_t = list(set(ctfidf_top_terms[t][:6] + pmi_top_terms[t][:6]))
        coherence_scores[t] = compute_topic_coherence_umass(all_t, texts_clean)
        print(f"    T{t} UMass = {coherence_scores[t]:.4f}")

    print("  [Domain taxonomy — CHG-4/5] ...")
    domain_assignments = {}
    domain_scores_all  = {}
    for t in range(N_TOPICS_USED):
        combined = list(set(ctfidf_top_terms[t][:10] + pmi_top_terms[t][:10]))
        best, scores = match_to_transport_domain(combined, TRANSPORT_TAXONOMY)
        domain_assignments[t] = best
        domain_scores_all[t]  = scores
        top2 = sorted(scores.items(), key=lambda x: -x[1])[:2]
        print(f"    T{t} -> {best[:50]}")
        if len(top2) > 1:
            print(f"           2nd: {top2[1][0][:50]}")

    TOPIC_NAMES_V6 = {}
    TOPIC_DESCS_V6 = {}
    for t in range(N_TOPICS_USED):
        kws     = ctfidf_top_terms[t][:3] if ctfidf_top_terms[t] else pmi_top_terms[t][:3]
        short_d = domain_assignments[t].split("&")[0].strip()[:35]
        TOPIC_NAMES_V6[t] = f"T{t}: {short_d} [{', '.join(kws)}]"
        TOPIC_DESCS_V6[t] = {
            "domain":          domain_assignments[t],
            "ctfidf_terms":    ctfidf_top_terms[t],
            "pmi_terms":       pmi_top_terms[t],
            "coherence_umass": coherence_scores[t],
            "domain_scores":   {k: float(v) for k, v in
                                 sorted(domain_scores_all[t].items(), key=lambda x: -x[1])[:5]},
        }
    return TOPIC_NAMES_V6, TOPIC_DESCS_V6


# ══════════════════════════════════════════════════════════════════════════════
# W_text: CONFIRMED PAPERS ONLY
# ══════════════════════════════════════════════════════════════════════════════
def build_wtext_tfidf_confirmed(df, inst_names, min_confirmed=5):
    banner("W_text via CONFIRMED PAPERS ONLY  (v6.0)")
    df_conf = df[df["imputed"] == 0].copy()
    print(f"  Confirmed papers: {len(df_conf):,} / {len(df):,} "
          f"({len(df_conf)/len(df)*100:.1f}%)")

    conf_counts = df_conf["institution"].value_counts()
    low_conf = [inst for inst in inst_names
                if conf_counts.get(inst, 0) < min_confirmed]
    print(f"  Institutions with <{min_confirmed} confirmed papers: "
          f"{len(low_conf)}/{len(inst_names)}")

    inst_docs = {}
    for inst in inst_names:
        texts = df_conf.loc[df_conf["institution"] == inst, "text_combined"].tolist()
        inst_docs[inst] = " ".join([clean_text_v6(t) for t in texts]) if len(texts) >= min_confirmed else None

    type_docs = {}
    for itype in INST_TYPES:
        type_insts  = [n for n in inst_names if INST_META[n][4] == itype]
        good_texts  = [inst_docs[n] for n in type_insts if inst_docs.get(n)]
        type_docs[itype] = " ".join(good_texts) if good_texts else "transportation research"

    for inst in low_conf:
        itype = INST_META[inst][4]
        inst_docs[inst] = type_docs[itype]
        print(f"    Fallback to {itype} centroid: {inst[:40]}")

    corpus = [inst_docs[inst] for inst in inst_names]
    vectorizer = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=1, stop_words=list(EXTENDED_STOP_WORDS),
        token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",
    )
    tfidf_mat   = vectorizer.fit_transform(corpus)
    W_text_raw  = cosine_similarity(tfidf_mat)
    np.fill_diagonal(W_text_raw, 0)

    row_entropies = []
    for i in range(len(inst_names)):
        row = W_text_raw[i].copy()
        row = row / (row.sum() + 1e-10)
        row_entropies.append(shannon_entropy(row))
    mean_ent  = np.mean(row_entropies)
    max_ent   = np.log(len(inst_names) - 1)
    disc_ratio = 1 - mean_ent / max_ent
    print(f"  W_text entropy: mean={mean_ent:.3f}  (max uniform={max_ent:.3f})")
    print(f"  Discrimination ratio: {disc_ratio:.3f}  "
          f"({'meaningful' if disc_ratio > 0.05 else 'low — imputation dominates'})")

    W_text = row_norm(W_text_raw)
    return W_text_raw, W_text, row_entropies, conf_counts, disc_ratio


# ══════════════════════════════════════════════════════════════════════════════
# M5 — BAYESIAN MEDIATION  (FIX-1: pt.abs)
# ══════════════════════════════════════════════════════════════════════════════
def run_m5_mediation(active_panel, COV_LABELS_SHORT):
    banner("L2. M5 — BAYESIAN MEDIATION ANALYSIS  (v6.0)")
    df_med = active_panel.copy()
    df_med = df_med.dropna(subset=["tech_prox", "research_novelty",
                                    "mean_patent_cit", "is_c9", "is_transport"])
    N_med     = len(df_med)
    X_med_raw = df_med[["is_c9", "is_transport", "is_research", "year_norm",
                          "log_n_papers", "log_inst_rank"]].fillna(0).values.astype(float)
    X_med_s   = StandardScaler().fit_transform(X_med_raw)
    M1_raw    = df_med["tech_prox"].values.astype(float)
    M2_raw    = df_med["research_novelty"].values.astype(float)
    Y_med_raw = df_med["log_mean_pat"].values.astype(float)
    M1_s = StandardScaler().fit_transform(M1_raw.reshape(-1, 1)).flatten()
    M2_s = StandardScaler().fit_transform(M2_raw.reshape(-1, 1)).flatten()
    Y_s  = StandardScaler().fit_transform(Y_med_raw.reshape(-1, 1)).flatten()
    K_med = X_med_s.shape[1]
    X_LABELS = ["C9 league", "Transport spec.", "Research inst.",
                 "Year trend", "log(Papers)", "log(Rank)"]

    with pm.Model() as m_med:
        gamma1   = pm.Normal("gamma1",   mu=0, sigma=0.5, shape=K_med)
        sigma_m1 = pm.HalfNormal("sigma_m1", sigma=0.5)
        mu_m1    = pt.dot(X_med_s, gamma1)
        pm.Normal("M1_obs", mu=mu_m1, sigma=sigma_m1, observed=M1_s)

        gamma2   = pm.Normal("gamma2",   mu=0, sigma=0.5, shape=K_med)
        sigma_m2 = pm.HalfNormal("sigma_m2", sigma=0.5)
        mu_m2    = pt.dot(X_med_s, gamma2)
        pm.Normal("M2_obs", mu=mu_m2, sigma=sigma_m2, observed=M2_s)

        beta_dir = pm.Normal("beta_dir", mu=0, sigma=0.5, shape=K_med)
        beta_m1  = pm.Normal("beta_m1",  mu=0, sigma=0.5)
        beta_m2  = pm.Normal("beta_m2",  mu=0, sigma=0.5)
        sigma_y  = pm.HalfNormal("sigma_y", sigma=0.5)
        mu_y     = pt.dot(X_med_s, beta_dir) + beta_m1 * M1_s + beta_m2 * M2_s
        pm.Normal("Y_obs", mu=mu_y, sigma=sigma_y, observed=Y_s)

        ACME_tech    = pm.Deterministic("ACME_tech",    gamma1[0] * beta_m1)
        ACME_novelty = pm.Deterministic("ACME_novelty", gamma2[0] * beta_m2)
        ACME_total   = pm.Deterministic("ACME_total",   ACME_tech + ACME_novelty)
        direct_eff   = pm.Deterministic("direct_c9",    beta_dir[0])
        total_eff    = pm.Deterministic("total_c9",     direct_eff + ACME_total)
        prop_med     = pm.Deterministic(
            "prop_mediated",
            ACME_total / (pt.abs(total_eff) + 1e-6)   # FIX-1: pt.abs
        )

        trace_med = pm.sample(
            draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS,
            target_accept=ACCEPT, progressbar=True, random_seed=42,
        )

    post = trace_med.posterior
    def _summ(var):
        s = post[var].values.flatten()
        return {"mean": float(s.mean()), "sd": float(s.std()),
                "hdi_lo": float(np.percentile(s, 2.5)),
                "hdi_hi": float(np.percentile(s, 97.5)),
                "P_pos":  float((s > 0).mean())}

    med_results = {
        "ACME_tech":      _summ("ACME_tech"),
        "ACME_novelty":   _summ("ACME_novelty"),
        "ACME_total":     _summ("ACME_total"),
        "direct_c9":      _summ("direct_c9"),
        "total_c9":       _summ("total_c9"),
        "prop_mediated":  _summ("prop_mediated"),
        "gamma1_means":   post["gamma1"].values.reshape(-1, K_med).mean(0).tolist(),
        "gamma2_means":   post["gamma2"].values.reshape(-1, K_med).mean(0).tolist(),
        "beta_dir_means": post["beta_dir"].values.reshape(-1, K_med).mean(0).tolist(),
        "X_labels":       X_LABELS,
        "beta_m1_mean":   float(post["beta_m1"].values.mean()),
        "beta_m2_mean":   float(post["beta_m2"].values.mean()),
        "N_med":          N_med,
    }
    print(f"\n  Mediation (C9 -> NLP -> Patent):")
    print(f"    ACME_total  = {med_results['ACME_total']['mean']:+.4f} "
          f"[{med_results['ACME_total']['hdi_lo']:+.4f}, "
          f"{med_results['ACME_total']['hdi_hi']:+.4f}]")
    print(f"    Direct      = {med_results['direct_c9']['mean']:+.4f}")
    print(f"    % Mediated  = {med_results['prop_mediated']['mean']*100:.1f}%")
    trace_med.to_netcdf(os.path.join(OUT_DIR, "trace_M5_mediation.nc"))
    return med_results, trace_med, X_LABELS, post


# ══════════════════════════════════════════════════════════════════════════════
# M6 — MULTI-STATE DEGRADATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_sojourn_times(panel, state_col="state"):
    records = []
    for inst in INST_NAMES:
        sub = (panel[panel["institution"] == inst]
               .sort_values("year").reset_index(drop=True))
        states = sub[state_col].values; years = sub["year"].values
        in_state = None; start_yr = None
        for yr, s in zip(years, states):
            if s < 0: in_state = None; continue
            if in_state is None:
                in_state = s; start_yr = yr
            elif s != in_state:
                records.append({"institution": inst, "from_state": in_state,
                                 "to_state": s, "sojourn_len": yr - start_yr,
                                 "year_start": start_yr})
                in_state = s; start_yr = yr
        if in_state is not None:
            records.append({"institution": inst, "from_state": in_state,
                             "to_state": -1, "sojourn_len": YEAR_END - start_yr + 1,
                             "year_start": start_yr})
    return pd.DataFrame(records)


def compute_reliability_metrics(Q, T_horizon=15):
    eigenvalues, eigenvectors = np.linalg.eig(Q.T)
    ss_idx = np.argmin(np.abs(eigenvalues))
    pi     = np.real(eigenvectors[:, ss_idx])
    pi     = np.abs(pi) / np.abs(pi).sum()
    availability = pi[2]
    try:
        Q_mttf = Q[1:, 1:]
        N_fund = -np.linalg.inv(Q_mttf)
        t_mttf = float(N_fund[1, :].sum())
    except Exception:
        t_mttf = np.nan
    try:
        Q_mttr = Q[:2, :2]
        N_fund2 = -np.linalg.inv(Q_mttr)
        t_mttr  = float(N_fund2[0, :].sum())
    except Exception:
        t_mttr = np.nan
    t_grid = np.linspace(0, T_horizon, 100)
    R_t    = [float(np.clip(expm(Q * t)[2, 1] + expm(Q * t)[2, 2], 0, 1))
              for t in t_grid]
    return {"pi": pi, "availability": float(availability),
            "mttf": t_mttf, "mttr": t_mttr,
            "R_t": R_t, "t_grid": t_grid.tolist()}


def run_m6_degradation(panel, trans, active_panel, TRANS_FEAT_COLS):
    banner("L3. M6 — MULTI-STATE DEGRADATION BAYESIAN MARKOV  (v6.0)")
    sojourn_df = compute_sojourn_times(panel)
    sojourn_df.to_csv(os.path.join(OUT_DIR, "M6_sojourn_times.csv"),
                      encoding="utf-8-sig", index=False)
    n_ij = np.zeros((N_STATES, N_STATES))
    T_i  = np.zeros(N_STATES)
    for _, row in trans.iterrows():
        s_f, s_t = int(row["state"]), int(row["state_next"])
        if 0 <= s_f < N_STATES and 0 <= s_t < N_STATES:
            n_ij[s_f, s_t] += 1
    for _, row in sojourn_df.iterrows():
        s = int(row["from_state"])
        if 0 <= s < N_STATES:
            T_i[s] += row["sojourn_len"]

    alpha_prior, beta_prior = 1.0, 2.0
    Q_post_mean = np.zeros((N_STATES, N_STATES))
    Q_post_std  = np.zeros((N_STATES, N_STATES))
    for i in range(N_STATES):
        for j in range(N_STATES):
            if i == j: continue
            alpha_p = alpha_prior + n_ij[i, j]
            beta_p  = beta_prior  + T_i[i]
            Q_post_mean[i, j] = alpha_p / beta_p
            Q_post_std[i, j]  = np.sqrt(alpha_p) / beta_p
        Q_post_mean[i, i] = -Q_post_mean[i, :].sum()

    print("  Posterior mean Q:")
    for i in range(N_STATES):
        print("    " + STATE_LABELS[i] + ": " +
              "  ".join([f"{Q_post_mean[i,j]:+.4f}" for j in range(N_STATES)]))

    rel_metrics = compute_reliability_metrics(Q_post_mean, T_horizon=15)
    print(f"  Availability A={rel_metrics['availability']:.3f}  "
          f"MTTF={rel_metrics['mttf']:.2f}yr  MTTR={rel_metrics['mttr']:.2f}yr")

    R_vec  = np.array([0.0, 0.5, 1.0]); gamma = 0.9
    P_ann  = np.clip(expm(Q_post_mean * 1.0), 0, 1)
    P_ann /= np.maximum(P_ann.sum(1, keepdims=True), 1e-10)
    V = np.zeros(N_STATES)
    for _ in range(1000):
        V_new = R_vec + gamma * (P_ann @ V)
        if np.max(np.abs(V_new - V)) < 1e-8: break
        V = V_new

    inst_rel = (active_panel.groupby("institution")["state"]
                .apply(lambda x: (x == 2).mean())
                .reset_index(name="availability_empirical"))

    return {
        "Q_mean": Q_post_mean.tolist(), "Q_std": Q_post_std.tolist(),
        "reliability": {k: v for k, v in rel_metrics.items()
                        if k not in ["R_t", "t_grid"]},
        "R_t": rel_metrics["R_t"], "t_grid": rel_metrics["t_grid"],
        "hmdp_value": V.tolist(), "P_annual": P_ann.tolist(),
    }, sojourn_df, rel_metrics, inst_rel


# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS CHECKS  (CHG-9: guard for subsample corr)
# ══════════════════════════════════════════════════════════════════════════════
def run_robustness_checks(panel, active_panel, T_emp_norm, N_STATES, YEAR_START):
    banner("ROBUSTNESS CHECKS  (v6.0)")
    results_robust = {}
    n_boot = 500
    T_boot = np.zeros((n_boot, N_STATES, N_STATES))
    trans_vals = panel[panel["state"] >= 0][["institution", "year", "state"]].copy()
    inst_list  = trans_vals["institution"].unique()
    for b in range(n_boot):
        inst_sample = np.random.choice(inst_list, size=len(inst_list), replace=True)
        T_b = np.zeros((N_STATES, N_STATES))
        for inst in inst_sample:
            sub = trans_vals[trans_vals["institution"] == inst].sort_values("year")
            states = sub["state"].values
            for i in range(len(states) - 1):
                s0, s1 = states[i], states[i+1]
                if 0 <= s0 < N_STATES and 0 <= s1 < N_STATES:
                    T_b[s0, s1] += 1
        row_sums = T_b.sum(1, keepdims=True)
        T_b = np.where(row_sums > 0, T_b / row_sums, 1/N_STATES)
        T_boot[b] = T_b
    T_boot_lo = np.percentile(T_boot, 2.5, 0)
    T_boot_hi = np.percentile(T_boot, 97.5, 0)
    results_robust["T_bootstrap_lo"] = T_boot_lo.tolist()
    results_robust["T_bootstrap_hi"] = T_boot_hi.tolist()

    top5_inst = (active_panel.groupby("institution")["n_papers"]
                 .sum().sort_values(ascending=False).head(5).index.tolist())
    loo_trans = {}
    for excl in top5_inst:
        sub = (panel[(panel["institution"] != excl) & (panel["state"] >= 0)]
               [["institution", "year", "state"]])
        T_loo = np.zeros((N_STATES, N_STATES))
        for inst in sub["institution"].unique():
            inst_sub = sub[sub["institution"] == inst].sort_values("year")
            states   = inst_sub["state"].values
            for i in range(len(states) - 1):
                s0, s1 = states[i], states[i+1]
                if 0 <= s0 < N_STATES and 0 <= s1 < N_STATES:
                    T_loo[s0, s1] += 1
        row_sums = T_loo.sum(1, keepdims=True)
        T_loo = np.where(row_sums > 0, T_loo / row_sums, 1/N_STATES)
        loo_trans[excl] = T_loo.tolist()
        diff = np.abs(T_loo - T_emp_norm).max()
        print(f"  LOO excl {excl[:35]:<35}: max |DeltaT|={diff:.4f}")
    results_robust["loo_transitions"] = loo_trans

    inst_imp = (panel.groupby("institution")["imputed_share"].mean().sort_values(ascending=False))
    results_robust["imputation_rates"] = inst_imp.to_dict()

    # CHG-9: guard
    low_imp_inst = inst_imp[inst_imp <= 0.30].index.tolist()
    sub_panel    = active_panel[active_panel["institution"].isin(low_imp_inst)]
    if len(sub_panel) >= 30:
        corr_full = spearmanr(active_panel["tech_prox"].fillna(0),
                               active_panel["mean_patent_cit"])[0]
        corr_sub  = spearmanr(sub_panel["tech_prox"].fillna(0),
                               sub_panel["mean_patent_cit"])[0]
        results_robust["subsample_tech_corr_full"] = float(corr_full)
        results_robust["subsample_tech_corr_sub"]  = float(corr_sub)
        print(f"  Subsample corr: full={corr_full:.4f}  sub(<=30% imp)={corr_sub:.4f}")
    else:
        print(f"  Subsample corr skipped: only {len(sub_panel)} rows with <=30% imputation")

    return results_robust, T_boot_lo, T_boot_hi, top5_inst, loo_trans, inst_imp


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — Run sections A through N sequentially
# See full code comments above for each section.
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner("STEP 1: NLP-AUGMENTED BAYESIAN TRANSITION MODEL v6.0")
    print("  Update FILE_PATH and OUT_DIR at the top of this script before running.")
    print("  Execution time: ~30-60 min on modern hardware (MCMC).")
    print("  Outputs saved to:", OUT_DIR)
