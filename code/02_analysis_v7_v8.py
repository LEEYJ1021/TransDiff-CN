# ===== 02_analysis_v7_v8.py : generated from a notebook; code cells only, outputs removed =====

# ---- cell 00 ----
# =====================================================================
# X1 | 절단 재현 + 솎아내기 실험 + 이항 GLM DID + 연도 제외(LOO)
# 입력: data_R1/papers_unique.parquet, 옛 33개 CSV
# =====================================================================
import warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, statsmodels.api as sm
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220); pd.set_option("display.max_columns", 50)

BASE = Path("~/Research/TransDiff-CN_R1").expanduser(); DATA = BASE / "data_R1"
OUT = BASE / "outputs_R1_v5"; OUT.mkdir(exist_ok=True)
OLD_CSV = BASE / "Transport_CN_Scholarly_Works.csv"
EVENT, SEED = 2013, 20261122
RNG = np.random.default_rng(SEED); RES = {}

pq, gz = DATA / "papers_unique.parquet", DATA / "papers_unique.csv.gz"
U = pd.read_parquet(pq) if pq.exists() else pd.read_csv(gz, low_memory=False)
U.columns = U.columns.str.strip()
U["year"] = pd.to_numeric(U["Publication Year"], errors="coerce")
U["cit"] = pd.to_numeric(U["Citing Patents Count"], errors="coerce")
U = U.dropna(subset=["year", "cit"]).reset_index(drop=True); U["year"] = U.year.astype(int)
U = U[U.year.between(2007, 2023)].reset_index(drop=True)
YEARS = sorted(U.year.unique())
low = lambda c: U[c].fillna("").astype(str).str.lower() if c in U.columns else pd.Series("", index=U.index)
fos = low("Fields of Study"); txt = low("Title") + " ; " + low("Keywords") + " ; " + fos

old = pd.read_csv(OLD_CSV, encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
old.columns = old.columns.str.strip()
old = old[old["Lens ID"].notna() & (old["Lens ID"].astype(str) != "Lens ID")].copy()
old["year"] = pd.to_numeric(old["Publication Year"], errors="coerce")
old["cit"] = pd.to_numeric(old["Citing Patents Count"], errors="coerce")
old = old.dropna(subset=["year", "cit"]).drop_duplicates("Lens ID"); old["year"] = old.year.astype(int)
old = old[old.year.between(2007, 2023)]

# ---------- T1. 선택 규칙 재현: A(인용≥1 조건)가 B의 압축 패턴을 만드는가 ----------
def pstats(d):
    lc = np.log1p(d.cit)
    return pd.Series(dict(n=len(d), sd_log=lc.std(ddof=1), share_eq1=(d.cit == 1).mean(), share_ge3=(d.cit >= 3).mean()))
T1 = pd.concat({"A_cited_only(89 inst)": U[U.cit >= 1].groupby("year").apply(pstats),
                "B_old33": old.groupby("year").apply(pstats)}, axis=1)
T1[("B_in_A", "coverage_of_A_cited")] = (U[(U.cit >= 1) & U["Lens ID"].isin(set(old["Lens ID"]))].groupby("year").size()
                                        / U[U.cit >= 1].groupby("year").size())
RES["T1_selection_replication"] = T1
print("\n### T1 인용≥1 조건만으로 B의 패턴이 재현되는지 (sd_log, share_eq1 열을 나란히 비교)")
print(T1.round(3).to_string())

# ---------- T2. 솎아내기(thinning) 실험: 노출만 줄여도 압축이 생기는가 ----------
base = U[U.year.between(2007, 2012)]; cits = base.cit.astype(int).values
def cohort_stats(c):
    c = np.asarray(c); k = c >= 1
    return dict(cited_share=k.mean(), share_eq1_among_cited=(c[k] == 1).mean() if k.any() else np.nan,
                sd_log_among_cited=np.log1p(c[k]).std(ddof=1) if k.sum() > 2 else np.nan)
rows = []
for q in (1.0, .75, .5, .35, .25, .15, .10):
    rr = pd.DataFrame([cohort_stats(RNG.binomial(cits, q)) for _ in range(30)]).mean()
    rows.append(dict(retention_q=q, **rr))
T2 = pd.DataFrame(rows).set_index("retention_q")
rec = U[U.year.between(2021, 2023)]
T2.loc["actual 2021-23"] = pd.Series(cohort_stats(rec.cit.values))
T2.loc["actual 2007-12"] = pd.Series(cohort_stats(cits))
RES["T2_thinning"] = T2
print("\n### T2 2007-12 코호트의 인용을 q 비율로 솎아낸 결과 vs 실제 2021-23 (q는 노출 비율의 대리, 시간 균질 가정)")
print(T2.round(3).to_string())

# ---------- T3. 이항 GLM DID + LOO ----------
BRI_RX = r"railway|high-speed rail|rail transit|\bports?\b|shipping|maritime|container|freight|logistics|belt and road|supply chain|intermodal|cross-border|corridor"
NAR_RX = r"railway|high-speed rail|rail transit|\bports?\b|shipping|maritime|freight|logistics"
NEV_RX = r"electric vehicle|battery|charging|new energy vehicle|hybrid vehicle|fuel cell"
PLC_RX = r"traffic signal|pavement|asphalt|driver|pedestrian|crash|road safety|traffic flow"
TRB = (r"transport|traffic|railway|\brail\b|vehicle|\broads?\b|highway|logistic|shipping|aviation|airport|"
       r"\bports?\b|freight|transit|pavement|automotive|driving|maritime|mobility|intermodal")
TRS = (r"traffic|railway|\brail\b|vehicle|highway|\broads?\b|aviation|airport|maritime|shipping|freight|"
       r"transit|public transport|pavement|automotive|autonomous driving")
POPS = {"all": np.ones(len(U), bool), "T_broad": fos.str.contains(TRB, regex=True).values,
        "T_strict": fos.str.contains(TRS, regex=True).values}
GRP = {"BRI_full[txt] (사전확정 주사양)": txt.str.contains(BRI_RX, regex=True).values,
       "BRI_narrow[txt]": txt.str.contains(NAR_RX, regex=True).values,
       "BRI_full[fos]": fos.str.contains(BRI_RX, regex=True).values,
       "BRI_narrow[fos]": fos.str.contains(NAR_RX, regex=True).values,
       "NEV[fos]": fos.str.contains(NEV_RX, regex=True).values,
       "Placebo_roadsafety[fos]": fos.str.contains(PLC_RX, regex=True).values}

def cell_counts(mpop, mg):
    d = pd.DataFrame({"y": U.year.values[mpop], "c": (U.cit.values[mpop] >= 1).astype(int), "g": mg[mpop]})
    t = d.groupby(["y", "g"]).c.agg(["sum", "size"]).unstack("g").reindex(YEARS).fillna(0)
    return t["sum"][True].values, t["size"][True].values, t["sum"][False].values, t["size"][False].values

def glm_did(cnt, years, drop=(), trend=False):
    cg, ng, cr, nr = cnt; rows = []
    for i, y in enumerate(YEARS):
        if y in drop or y not in years: continue
        for g, (c, n) in enumerate([(cr[i], nr[i]), (cg[i], ng[i])]):
            if n > 0: rows.append((y, g, c, n - c))
    d = pd.DataFrame(rows, columns=["y", "g", "s", "f"])
    X = pd.get_dummies(d.y, dtype=float); X["g"] = d.g; X["gpost"] = d.g * (d.y >= EVENT)
    if trend: X["gtrend"] = d.g * (d.y - EVENT)
    m = sm.GLM(d[["s", "f"]].values, X.values.astype(float), family=sm.families.Binomial()).fit(scale="X2")
    j = list(X.columns).index("gpost")
    return m.params[j], m.bse[j]

SPECS = {"all_years": (list(YEARS), False), "all_years+group_trend": (list(YEARS), True),
         "symmetric(2008-2017)": (list(range(2008, 2018)), False)}
rows, LOO = [], {}
for pn, pm in POPS.items():
    for gn, gm in GRP.items():
        cnt = cell_counts(pm, gm); cg, ng = cnt[0], cnt[1]
        pre = [y for y in YEARS if y < EVENT]; post = [y for y in YEARS if y >= EVENT]
        ipre = [YEARS.index(y) for y in pre]; ipost = [YEARS.index(y) for y in post]
        for sn, (yrs, tr) in SPECS.items():
            try: b, se = glm_did(cnt, yrs, trend=tr)
            except Exception: continue
            loo = {}
            for y in yrs:
                try: loo[y] = glm_did(cnt, yrs, drop=(y,), trend=tr)[0]
                except Exception: pass
            L = pd.Series(loo); LOO[(pn, gn, sn)] = L
            rows.append(dict(population=pn, target=gn, spec=sn, delta_logOR=b, se_quasi=se, z=b / se,
                             loo_min=L.min(), loo_max=L.max(), loo_sign_stable=bool((np.sign(L) == np.sign(b)).all()),
                             most_influential_year=int((L - b).abs().idxmax()),
                             cited_pre=int(cg[ipre].sum()), cited_post=int(cg[ipost].sum()),
                             papers_group=int(ng.sum())))
T3 = pd.DataFrame(rows); RES["T3_binomial_DID_LOO"] = T3
print("\n### T3 이항 GLM(연도 고정효과, quasi-binomial) DID: delta>0이면 대조 대비 인용 확률비 증가")
print(T3.round(3).to_string())
print("※ cited_pre/cited_post가 작으면(수십 편) 검정력이 낮음. loo_sign_stable=False면 한 해에 부호가 좌우됨. "
      "se는 quasi-binomial이며 연도 간 상관을 반영하지 않아 낙관적일 수 있음")

# ---------- 그림 ----------
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
a = T1.xs("sd_log", axis=1, level=1)
ax[0].plot(a.index, a.iloc[:, 0] / a.iloc[:, 0].iloc[0], "o-", ms=3, label="A: cited-only (89 inst)")
ax[0].plot(a.index, a.iloc[:, 1] / a.iloc[:, 1].iloc[0], "s--", ms=3, label="B: old 33 inst")
ax[0].set_title("(a) Paper-level SD under the same selection rule (2007=1)", loc="left", fontweight="bold"); ax[0].legend(fontsize=7, frameon=False)
ax[1].plot(T2.index[:7].astype(float), T2.iloc[:7].share_eq1_among_cited, "o-", label="thinned 2007-12 cohort")
ax[1].axhline(T2.loc["actual 2021-23", "share_eq1_among_cited"], c="r", ls="--", label="actual 2021-23")
ax[1].set_xlabel("retention q"); ax[1].set_title("(b) Share with exactly 1 citation | cited", loc="left", fontweight="bold"); ax[1].legend(fontsize=7, frameon=False)
sub = T3[(T3.population == "T_broad") & (T3.spec == "all_years")]
ax[2].errorbar(sub.delta_logOR, range(len(sub)), xerr=[sub.delta_logOR - sub.loo_min, sub.loo_max - sub.delta_logOR], fmt="o", capsize=3)
ax[2].set_yticks(range(len(sub))); ax[2].set_yticklabels(sub.target, fontsize=6); ax[2].axvline(0, c="k", lw=.8)
ax[2].set_title("(c) DID with leave-one-year-out range [T_broad]", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig(OUT / "fig_X1.png", dpi=300, bbox_inches="tight"); plt.savefig(OUT / "fig_X1.pdf", bbox_inches="tight"); plt.show()

for k, v in RES.items(): v.to_csv(OUT / f"{k}.csv", encoding="utf-8-sig")
print("저장:", OUT)

# ---- cell 01 ----
# =====================================================================
# X2 | 89개 기관 패널(전체/인용 건수) -> 기관 간 이질성(DL τ), 옛33 vs 나머지, (선택) Moran
# 입력: panel_txt/{year}_{all|cited}.txt, candidate_institutions_filled.csv, 옛 33개 CSV
# =====================================================================
import re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
BASE = Path("~/Research/TransDiff-CN_R1").expanduser(); PAN = BASE / "panel_txt"
OUT = BASE / "outputs_R1_v5"; OUT.mkdir(exist_ok=True)
CAND = BASE / "candidate_institutions_filled.csv"; OLD_CSV = BASE / "Transport_CN_Scholarly_Works.csv"
COORD = BASE / "inst_coords_89.csv"           # 선택: 열 Institution,lat,lon (없으면 템플릿 생성)
YEARS = list(range(2007, 2024)); MIN_N = 20; B = 999; SEED = 20261122
PER = {"2007-12": (2007, 2012), "2013-19": (2013, 2019), "2020-23": (2020, 2023)}
RNG = np.random.default_rng(SEED)
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()

cand = pd.read_csv(CAND, encoding="utf-8-sig")["name"].dropna().astype(str).str.strip().drop_duplicates().tolist()
cmap = {norm(c): c for c in cand}
def parse(text):
    t = re.sub(r"Load More|View all|More", " ", str(text)); t = re.sub(r"\s+", " ", t); out, pos = {}, 0
    for m in re.finditer(r"\(\s*([\d,]+)\s*\)", t):
        nm = t[pos:m.start()].strip(); pos = m.end()
        if nm: out[norm(nm)] = int(m.group(1).replace(",", ""))
    return out

N = pd.DataFrame(np.nan, index=cand, columns=YEARS); C = N.copy(); log = []
for y in YEARS:
    for kind, M in (("all", N), ("cited", C)):
        f = PAN / f"{y}_{kind}.txt"
        if not f.exists(): log.append((y, kind, "파일 없음", 0)); continue
        p = parse(f.read_text(encoding="utf-8", errors="replace")); hit = 0
        for k, v in p.items():
            if k in cmap: M.loc[cmap[k], y] = v; hit += 1
        log.append((y, kind, "ok", hit))
L = pd.DataFrame(log, columns=["year", "kind", "status", "matched_candidates"])
print(L.pivot(index="year", columns="kind", values="matched_candidates").to_string())
if (L.status != "ok").any(): raise SystemExit("파일이 없는 연도가 있습니다: " + str(L[L.status != "ok"][["year", "kind"]].values.tolist()))
N, C = N.fillna(0), C.fillna(0)               # 패널에 없던 기관=0 (Load More를 끝까지 펼친 경우에만 타당)
print("※ 인용 건수가 전체 건수보다 큰 셀:", int((C > N).sum().sum()), "(0이어야 정상)")

# ---- 옛 33개와 검증 ----
old = pd.read_csv(OLD_CSV, encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
old.columns = old.columns.str.strip(); old = old[old["Lens ID"].notna() & (old["Lens ID"].astype(str) != "Lens ID")]
old["year"] = pd.to_numeric(old["Publication Year"], errors="coerce"); old = old.dropna(subset=["year"])
old["Institution"] = old["Institution"].map(lambda s: cmap.get(norm(s), cmap.get(norm(re.sub(r"\s*\(.*\)$", "", str(s))))))
old = old.dropna(subset=["Institution"]).drop_duplicates(["Lens ID", "Institution"])
OLD33 = sorted(old.Institution.unique())
Bc = old.groupby(["Institution", old.year.astype(int)]).size().unstack().reindex(index=OLD33, columns=YEARS).fillna(0)
Cc = C.loc[OLD33]
print(f"\n옛 파일 매칭 기관 {len(OLD33)}개 | 패널 인용 건수 합 vs 옛 파일 합: {int(Cc.values.sum())} vs {int(Bc.values.sum())} "
      f"| 셀 상관 {np.corrcoef(Cc.values.ravel(), Bc.values.ravel())[0, 1]:.3f}")
print("※ 패널(추출 시점 최신) ≥ 옛 파일이어야 함. 크게 다르면 기관명 매칭이나 정의 차이를 점검")

# ---- DerSimonian-Laird: 기관 간 진짜 이질성 τ와 표집잡음 분리 ----
def dl(c, n, min_n=MIN_N):
    k_ = n >= min_n; c, n = c[k_], n[k_]; k = len(c)
    if k < 8: return None
    pbar = c.sum() / n.sum(); p = c / n; w = n / (pbar * (1 - pbar))
    Q = float((w * (p - pbar) ** 2).sum()); df = k - 1
    tau2 = max(0.0, (Q - df) / (w.sum() - (w ** 2).sum() / w.sum()))
    return dict(k=k, pbar=pbar, Q=Q, p_hetero=stats.chi2.sf(Q, df), tau=np.sqrt(tau2), cv_true=np.sqrt(tau2) / pbar,
                I2=max(0.0, (Q - df) / Q) if Q > 0 else 0.0)

def pooled(M, a, b): return M.loc[:, a:b].sum(axis=1).values
groups = {"all89": cand, "old33": OLD33, "other56": [c for c in cand if c not in set(OLD33)]}
rows = []
for gn, names in groups.items():
    for pn, (a, b) in PER.items():
        r = dl(pooled(C.loc[names], a, b).astype(float), pooled(N.loc[names], a, b).astype(float))
        if r: rows.append(dict(group=gn, period=pn, **r))
X21 = pd.DataFrame(rows); X21.to_csv(OUT / "X2_1_heterogeneity.csv", index=False)
print("\n### X2-1 기관 간 이질성 (cv_true = 표집잡음 제거 후 σ/평균; I2=이질성 비중)"); print(X21.round(4).to_string(index=False))

# 부트스트랩: 기관을 재표집해 CV 비율의 구간 (기간 간 비교)
def boot_ratio(names, p1, p2):
    c1, n1 = pooled(C.loc[names], *PER[p1]).astype(float), pooled(N.loc[names], *PER[p1]).astype(float)
    c2, n2 = pooled(C.loc[names], *PER[p2]).astype(float), pooled(N.loc[names], *PER[p2]).astype(float)
    ok = (n1 >= MIN_N) & (n2 >= MIN_N); idx = np.where(ok)[0]; out = []
    for _ in range(B):
        s = RNG.choice(idx, len(idx)); a, b = dl(c1[s], n1[s]), dl(c2[s], n2[s])
        if a and b and a["cv_true"] > 0: out.append(b["cv_true"] / a["cv_true"])
    return len(idx), np.percentile(out, [2.5, 50, 97.5]) if out else [np.nan] * 3
rows = []
for gn in ("all89", "old33", "other56"):
    for p1, p2 in (("2007-12", "2013-19"), ("2013-19", "2020-23"), ("2007-12", "2020-23")):
        k, q = boot_ratio(groups[gn], p1, p2)
        rows.append(dict(group=gn, compare=f"{p2}/{p1}", n_inst_balanced=k, cv_ratio_q025=q[0], median=q[1], q975=q[2]))
X22 = pd.DataFrame(rows); X22.to_csv(OUT / "X2_2_cv_ratio_boot.csv", index=False)
print("\n### X2-2 이질성(CV) 비율. 구간이 1을 포함하면 '수렴'으로 읽을 수 없음"); print(X22.round(3).to_string(index=False))

# ---- 성과 기반 선정 점검: 옛 33개 vs 나머지 56개 ----
sh = pd.DataFrame({g: C.loc[n].sum() / N.loc[n].sum() for g, n in groups.items() if g != "all89"})
sh["old33_share_of_cited_papers"] = C.loc[OLD33].sum() / C.sum()
sh["papers_old33_share"] = N.loc[OLD33].sum() / N.sum()
print("\n### X2-3 인용 비중과 옛 33개가 차지하는 비중 (연도별)"); print(sh.round(4).to_string())
sh.to_csv(OUT / "X2_3_old33_vs_other56.csv")

# ---- 순위 지속성: 사전(2007-12) 순위가 이후에도 유지되는가 (평균회귀 점검) ----
def p_per(per): return pooled(C, *PER[per]) / np.maximum(pooled(N, *PER[per]), 1)
ok = (pooled(N, *PER["2007-12"]) >= MIN_N) & (pooled(N, *PER["2013-19"]) >= MIN_N) & (pooled(N, *PER["2020-23"]) >= MIN_N)
for a, b in (("2007-12", "2013-19"), ("2013-19", "2020-23"), ("2007-12", "2020-23")):
    r, pv = stats.spearmanr(p_per(a)[ok], p_per(b)[ok]); print(f"순위 지속성 Spearman {a} vs {b}: rho={r:.3f}, p={pv:.3f}, n={ok.sum()}")

# ---- 선택: 공간 의존성 (Pearson 잔차 Moran, 같은 도시 링크 제외) ----
if not COORD.exists():
    pd.DataFrame({"Institution": cand, "city": "", "lat": "", "lon": ""}).to_csv(BASE / "inst_coords_89_TEMPLATE.csv", index=False)
    print("\n[공간 생략] inst_coords_89.csv 없음. TEMPLATE을 채워(Institution,city,lat,lon) 저장 후 재실행")
else:
    co = pd.read_csv(COORD).dropna(subset=["lat", "lon"]).set_index("Institution")
    def hav(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        a = np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la)[:, None] * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(a))
    def moran_perm(z, W):
        z = z - z.mean(); n = len(z); S0 = W.sum()
        I = n / S0 * (z @ W @ z) / (z @ z)
        Z = np.array([RNG.permutation(z) for _ in range(4999)])
        Ip = n / S0 * np.einsum("ri,ij,rj->r", Z, W, Z) / (z @ z)
        return I, (1 + np.sum(np.abs(Ip - Ip.mean()) >= abs(I - Ip.mean()))) / 5000
    rows = []
    for pn in ("2007-12", "2013-19", "2020-23"):
        c, n = pooled(C, *PER[pn]).astype(float), pooled(N, *PER[pn]).astype(float)
        k = (n >= MIN_N) & pd.Series(cand).isin(co.index).values
        pbar = c[k].sum() / n[k].sum(); z = (c[k] / n[k] - pbar) / np.sqrt(pbar * (1 - pbar) / n[k])   # Pearson 잔차
        nm = np.array(cand)[k]; D = hav(co.loc[nm, "lat"].values.astype(float), co.loc[nm, "lon"].values.astype(float))
        for wn, W in (("inv_dist floor25", 1 / np.maximum(D, 25)),
                      ("inv_dist excl same-city(<30km)", np.where(D < 30, 0, 1 / np.maximum(D, 25)))):
            np.fill_diagonal(W, 0); W = W / W.sum(1, keepdims=True).clip(min=1e-12)
            I, p = moran_perm(z, W); rows.append(dict(period=pn, weights=wn, n=int(k.sum()), moran_I=I, p=p))
    X24 = pd.DataFrame(rows); X24.to_csv(OUT / "X2_4_moran_pearson_residuals.csv", index=False)
    print("\n### X2-4 Moran (Pearson 잔차)"); print(X24.round(4).to_string(index=False))
print("저장:", OUT)

# ---- cell 02 ----
# =====================================================================
# X3 | 기관 패널 전처리 + 이질성(DL) 분석 + 중국 지리 시각화 (파일이 일부만 있어도 동작)
# 입력 : panel_txt/{year}_{all|cited}.txt, candidate_institutions_filled.csv,
#        Transport_CN_Scholarly_Works.csv(옛 33개), gadm41_CHN_1.json, (선택) inst_coords_89.csv
# 필요 : pandas numpy scipy matplotlib geopandas
# =====================================================================
import re, difflib, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy import stats
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40)

# ----------------------------- 설정 ---------------------------------
BASE   = Path("~/Research/TransDiff-CN_R1").expanduser()
PAN    = BASE / "panel_txt"; PAN.mkdir(exist_ok=True)
OUT    = BASE / "outputs_R1_v6"; OUT.mkdir(exist_ok=True)
CAND   = BASE / "candidate_institutions_filled.csv"
OLD    = BASE / "Transport_CN_Scholarly_Works.csv"
GEOJSON= BASE / "gadm41_CHN_1.json"
COORD  = BASE / "inst_coords_89.csv"            # 열: Institution,city,lat,lon (선택)
MIN_N_PERIOD = 100     # 기간 합산 논문 수 최소(기관 포함 기준). 비율이 작아 20은 정규근사가 부정확
MIN_N_YEAR   = 40      # 연도별 이질성 곡선용 최소 논문 수
B, SEED = 999, 20261122
RNG = np.random.default_rng(SEED)
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
strip_par = lambda s: re.sub(r"\s*\([^)]*\)\s*$", "", str(s)).strip()
RES = {}

# ========================== 0. 후보·옛 파일 로드 ======================
cand = (pd.read_csv(CAND, encoding="utf-8-sig")["name"].dropna().astype(str).str.strip().drop_duplicates().tolist())
cmap = {norm(c): c for c in cand}
cmap_np = {norm(strip_par(c)): c for c in cand}
print(f"후보 기관 {len(cand)}개")

old = pd.read_csv(OLD, encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
old.columns = old.columns.str.strip()
old = old[old["Lens ID"].notna() & (old["Lens ID"].astype(str) != "Lens ID")].copy()
old["year"] = pd.to_numeric(old["Publication Year"], errors="coerce")
old["cit"] = pd.to_numeric(old["Citing Patents Count"], errors="coerce")
old = old.dropna(subset=["year", "Institution"]); old["year"] = old.year.astype(int)
def to_cand(s):
    k = norm(s); return cmap.get(k) or cmap_np.get(norm(strip_par(s))) or cmap_np.get(k)
old["Institution"] = old["Institution"].map(to_cand)
old = old.dropna(subset=["Institution"]).drop_duplicates(["Lens ID", "Institution"])
OLD33 = sorted(old.Institution.unique())
print(f"옛 파일에서 후보와 매칭된 기관 {len(OLD33)}개")

# ========================== 1. 패널 파일 점검 =========================
files = {}
for f in PAN.glob("*.txt"):
    m = re.fullmatch(r"(\d{4})_(all|cited)\.txt", f.name.strip())
    if m: files[(int(m.group(1)), m.group(2))] = f
have = sorted({y for (y, k) in files if (y, "all") in files and (y, "cited") in files})
chk = pd.DataFrame([dict(year=y, all_file=(y, "all") in files, cited_file=(y, "cited") in files) for y in range(2007, 2024)])
chk.to_csv(BASE / "panel_txt_CHECKLIST.csv", index=False)
print("\n[패널 파일 현황]"); print(chk.to_string(index=False))
print(f"→ 분석 가능 연도(all+cited 둘 다): {have}")

# ========================== 2. 파서 ==================================
def parse_panel(text):
    t = text.replace("\u00a0", " ")
    t = re.sub(r"\b(Load More|Show More|View all)\b", "\n", t, flags=re.I)
    # 형식 1: 이름 (1,234) — 숫자 괄호 직전까지를 이름으로
    flat = re.sub(r"\s+", " ", t); r1, pos = [], 0
    for m in re.finditer(r"\(\s*([\d,]+)\s*\)", flat):
        nm = flat[pos:m.start()].strip(); pos = m.end()
        if nm: r1.append((nm, int(m.group(1).replace(",", ""))))
    # 형식 2: 이름 줄 / 숫자 줄
    r2, prev = [], None
    for ln in [x.strip() for x in t.splitlines() if x.strip()]:
        if re.fullmatch(r"[\d,]+", ln):
            if prev: r2.append((prev, int(ln.replace(",", "")))); prev = None
        else: prev = ln
    return r1 if len(r1) >= len(r2) else r2

def match_name(nm):
    k = norm(nm)
    if k in cmap: return cmap[k], "exact"
    if norm(strip_par(nm)) in cmap_np: return cmap_np[norm(strip_par(nm))], "no_paren"
    g = difflib.get_close_matches(k, list(cmap), n=1, cutoff=0.92)
    return (cmap[g[0]], "fuzzy") if g else (None, "none")

# ========================== 3. 패널 구축 ==============================
N = pd.DataFrame(np.nan, index=cand, columns=have); C = N.copy(); plog, unmatched = [], []
for y in have:
    for kind, M in (("all", N), ("cited", C)):
        rows = parse_panel(files[(y, kind)].read_text(encoding="utf-8", errors="replace"))
        hit = fz = 0
        for nm, v in rows:
            c, how = match_name(nm)
            if c is None: unmatched.append((y, kind, nm, v)); continue
            M.loc[c, y] = max(v, M.loc[c, y]) if not np.isnan(M.loc[c, y]) else v
            hit += 1; fz += how == "fuzzy"
        plog.append(dict(year=y, kind=kind, rows_parsed=len(rows), matched=hit, fuzzy=fz))
PL = pd.DataFrame(plog); RES["panel_parse_log"] = PL
if have:
    print("\n[파싱 로그] matched가 연도마다 크게 다르면 Load More를 덜 눌렀을 가능성"); print(PL.pivot(index="year", columns="kind", values="matched").to_string())
    um = pd.DataFrame(unmatched, columns=["year", "kind", "name", "n"])
    print("\n[후보에 없는 소속 상위 — 표기 변형이면 candidate 이름과 비교]")
    print(um.groupby("name").n.sum().sort_values(ascending=False).head(15).to_string())
    um.to_csv(OUT / "unmatched_names.csv", index=False)
    bad = int((C > N).sum().sum()); print(f"\n인용 건수 > 전체 건수 셀: {bad} (0이어야 정상)")
    N, C = N.fillna(0), C.fillna(0)   # 결측=0. 아래 분석은 MIN_N 필터로 0셀을 자동 제외

# ========================== 4. 좌표 ==================================
K33 = {  # 앞서 사용한 33개 좌표(근삿값, 검증 필요): city, lat, lon
 "Chinese Academy of Sciences":("Beijing",39.98,116.33),"Tsinghua University":("Beijing",40.00,116.33),
 "Peking University":("Beijing",39.99,116.31),"Beihang University":("Beijing",39.98,116.35),
 "Beijing Institute of Technology":("Beijing",39.96,116.31),"Beijing Jiaotong University":("Beijing",39.95,116.34),
 "University of Chinese Academy of Sciences":("Beijing",39.91,116.24),"Beijing University of Technology":("Beijing",39.87,116.48),
 "Tianjin University":("Tianjin",39.11,117.17),"Harbin Institute of Technology":("Harbin",45.75,126.68),
 "Jilin University":("Changchun",43.83,125.29),"Dalian University of Technology":("Dalian",38.89,121.54),
 "Shandong University":("Jinan",36.67,117.07),"Xi'an Jiaotong University":("Xi'an",34.25,108.98),
 "Sichuan University":("Chengdu",30.63,104.08),"Southwest Jiaotong University":("Chengdu",30.70,104.00),
 "Chongqing University":("Chongqing",29.57,106.47),"Central South University":("Changsha",28.17,112.93),
 "University of Science and Technology of China":("Hefei",31.84,117.26),"Nanjing University":("Nanjing",32.06,118.78),
 "Southeast University":("Nanjing",32.06,118.80),"Tongji University":("Shanghai",31.28,121.50),
 "Fudan University":("Shanghai",31.30,121.50),"Shanghai Jiao Tong University":("Shanghai",31.03,121.44),
 "Shanghai University":("Shanghai",31.32,121.40),"Zhejiang University":("Hangzhou",30.30,120.09),
 "Wuhan University":("Wuhan",30.54,114.36),"Wuhan University of Technology":("Wuhan",30.51,114.33),
 "Huazhong University of Science and Technology":("Wuhan",30.51,114.41),"South China University of Technology":("Guangzhou",23.15,113.34),
 "Sun Yat-sen University":("Guangzhou",23.10,113.30),"Shenzhen University":("Shenzhen",22.53,113.93),
 "Soochow University":("Suzhou",31.30,120.63)}
geo = pd.DataFrame({"Institution": cand})
k33 = {norm(strip_par(k)): v for k, v in K33.items()}
geo[["city", "lat", "lon"]] = [k33.get(norm(strip_par(c)), (None, np.nan, np.nan)) for c in cand]
if COORD.exists():
    ov = pd.read_csv(COORD).set_index("Institution")
    for i, r in geo.iterrows():
        if r.Institution in ov.index and pd.notna(ov.loc[r.Institution, "lat"]):
            geo.loc[i, ["city", "lat", "lon"]] = [ov.loc[r.Institution].get("city"), float(ov.loc[r.Institution, "lat"]), float(ov.loc[r.Institution, "lon"])]
miss = geo[geo.lat.isna()]
if len(miss):
    miss.to_csv(BASE / "inst_coords_89_TEMPLATE.csv", index=False)
    print(f"\n[좌표] {len(geo) - len(miss)}/{len(geo)}개 보유. 나머지 {len(miss)}개는 inst_coords_89_TEMPLATE.csv(Institution,city,lat,lon)를 채워 inst_coords_89.csv로 저장")
geo = geo.set_index("Institution")

# ========================== 5. 지도 도구 =============================
import geopandas as gpd
AEA = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs"
prov = None
if GEOJSON.exists():
    prov = gpd.read_file(GEOJSON); prov = (prov.set_crs(4326) if prov.crs is None else prov).to_crs(AEA)
    prov["geometry"] = prov.geometry.simplify(2000); prov = prov.dissolve(by="NAME_1", as_index=False)
else:
    print("[지도] 경계 파일 없음 → 점만 그림:", GEOJSON)

def xy(names):
    g = geo.loc[[n for n in names if n in geo.index and pd.notna(geo.loc[n, "lat"])]].copy()
    p = gpd.GeoSeries(gpd.points_from_xy(g.lon, g.lat), crs=4326).to_crs(AEA)
    g["px"], g["py"] = p.x.values, p.y.values; return g
def base_map(ax, g, pad=7e5, full=False):
    if prov is not None: prov.plot(ax=ax, color="#f4f2ec", edgecolor="#bdb9ae", linewidth=.3, zorder=0)
    if full and prov is not None:
        b = prov.total_bounds; ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
    else: ax.set_xlim(g.px.min() - pad, g.px.max() + pad); ax.set_ylim(g.py.min() - pad, g.py.max() + pad)
    ax.set_aspect("equal"); ax.set_axis_off()
def beijing_inset(ax, g, draw):
    bj = g[g.city == "Beijing"]
    if len(bj) < 2: return
    ia = ax.inset_axes([0.02, 0.58, 0.30, 0.38]); base_map(ia, bj, pad=6e4)
    draw(ia, bj); ia.set_title("Beijing", fontsize=7, pad=2); ia.set_axis_on(); ia.set_xticks([]); ia.set_yticks([])
    cx, cy = bj.px.mean(), bj.py.mean(); m = len(bj)
    for j, (n, r) in enumerate(bj.sort_values("py").iterrows()):
        a = 2 * np.pi * j / m; ia.annotate(re.sub(r"University|Institute|of|and|the", "", n).strip()[:14], (r.px, r.py),
            (cx + 5e4 * np.cos(a), cy + 5e4 * np.sin(a)), fontsize=5, ha="center", arrowprops=dict(arrowstyle="-", lw=.3, color="#777"))
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})

# ========================== 6. 패널 분석 ==============================
def dl(c, n, min_n):
    k_ = n >= min_n; c, n = c[k_], n[k_]; k = len(c)
    if k < 8 or c.sum() == 0: return None
    pbar = c.sum() / n.sum(); p = c / n; w = n / (pbar * (1 - pbar))
    Q = float((w * (p - pbar) ** 2).sum()); df = k - 1
    tau2 = max(0.0, (Q - df) / (w.sum() - (w ** 2).sum() / w.sum()))
    return dict(k=k, pbar=pbar, Q=Q, p_hetero=stats.chi2.sf(Q, df), tau=np.sqrt(tau2), cv_true=np.sqrt(tau2) / pbar,
                I2=max(0.0, (Q - df) / Q) if Q > 0 else 0.0)

if len(have) >= 3:
    yrs = have
    PER = {"2007-12": [y for y in yrs if y <= 2012], "2013-19": [y for y in yrs if 2013 <= y <= 2019], "2020-23": [y for y in yrs if y >= 2020]}
    PER = {k: v for k, v in PER.items() if v}
    print("\n기간 구성(있는 연도 기준):", PER)
    pooled = lambda M, ys: M[ys].sum(axis=1).values.astype(float)
    groups = {"all89": cand, "old33": OLD33, "other56": [c for c in cand if c not in set(OLD33)]}

    rows = []
    for gn, names in groups.items():
        for pn, ys in PER.items():
            r = dl(pooled(C.loc[names], ys), pooled(N.loc[names], ys), MIN_N_PERIOD)
            if r: rows.append(dict(group=gn, period=pn, years=len(ys), **r))
    X1 = pd.DataFrame(rows); RES["X3_1_heterogeneity"] = X1
    print("\n### X3-1 기관 간 이질성 (cv_true = 표집잡음 제거 후 σ/평균, I2 = 이질성 비중)"); print(X1.round(4).to_string(index=False))

    def boot_ratio(names, p1, p2):
        c1, n1 = pooled(C.loc[names], PER[p1]), pooled(N.loc[names], PER[p1])
        c2, n2 = pooled(C.loc[names], PER[p2]), pooled(N.loc[names], PER[p2])
        idx = np.where((n1 >= MIN_N_PERIOD) & (n2 >= MIN_N_PERIOD))[0]; out = []
        for _ in range(B):
            s = RNG.choice(idx, len(idx)); a, b = dl(c1[s], n1[s], MIN_N_PERIOD), dl(c2[s], n2[s], MIN_N_PERIOD)
            if a and b and a["cv_true"] > 0: out.append(b["cv_true"] / a["cv_true"])
        return len(idx), (np.percentile(out, [2.5, 50, 97.5]) if len(out) > 50 else [np.nan] * 3)
    rows = []; pk = list(PER)
    for gn in groups:
        for i, j in [(a, b) for a in range(len(pk)) for b in range(a + 1, len(pk))]:
            k_, q = boot_ratio(groups[gn], pk[i], pk[j])
            rows.append(dict(group=gn, compare=f"{pk[j]}/{pk[i]}", n_inst=k_, q025=q[0], median=q[1], q975=q[2]))
    X2 = pd.DataFrame(rows); RES["X3_2_cv_ratio"] = X2
    print("\n### X3-2 이질성(CV) 비율 부트스트랩. 구간이 1을 포함하면 '수렴'으로 읽을 수 없음"); print(X2.round(3).to_string(index=False))

    ycv = []
    for y in yrs:
        r = dl(C[y].values.astype(float), N[y].values.astype(float), MIN_N_YEAR)
        if r: ycv.append(dict(year=y, **r))
    YCV = pd.DataFrame(ycv).set_index("year"); RES["X3_3_cv_by_year"] = YCV
    print("\n### X3-3 연도별 이질성 (k=포함 기관 수; k가 변하면 구성 효과 주의)"); print(YCV.round(4).to_string())

    sh = pd.DataFrame({"cited_share_all89": C.sum() / N.sum(), "cited_share_old33": C.loc[OLD33].sum() / N.loc[OLD33].sum(),
                       "old33_share_of_cited": C.loc[OLD33].sum() / C.sum(), "old33_share_of_papers": N.loc[OLD33].sum() / N.sum()})
    RES["X3_4_old33_coverage"] = sh; print("\n### X3-4 옛 33개의 대표성"); print(sh.round(4).to_string())

    rate = lambda ys: pooled(C, ys) / np.maximum(pooled(N, ys), 1)
    okp = np.all([pooled(N, ys) >= MIN_N_PERIOD for ys in PER.values()], axis=0)
    for i in range(len(pk)):
        for j in range(i + 1, len(pk)):
            r, p = stats.spearmanr(rate(PER[pk[i]])[okp], rate(PER[pk[j]])[okp])
            print(f"순위 지속성 {pk[i]} vs {pk[j]}: rho={r:.3f}, p={p:.3f}, n={okp.sum()}")

    # 옛 파일과 검증(패널 인용 건수 ≥ 옛 파일이어야 함)
    Bc = old.groupby(["Institution", "year"]).size().unstack().reindex(index=OLD33, columns=yrs).fillna(0)
    Cc = C.loc[OLD33]
    print(f"\n[검증] 패널 인용 건수 합 {int(Cc.values.sum())} vs 옛 파일 {int(Bc.values.sum())}, 셀 상관 {np.corrcoef(Cc.values.ravel(), Bc.values.ravel())[0, 1]:.3f}")

    # ---------- 그림: 이질성/구성 ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(YCV.index, YCV.cv_true, "o-", ms=3, label="CV (noise-removed)")
    a2 = ax[0].twinx(); a2.plot(YCV.index, YCV.k, "s:", c="gray", ms=3, label="institutions included"); a2.set_ylabel("k")
    ax[0].axvline(2013, c="k", ls=":", lw=1); ax[0].set_title("(a) Between-institution heterogeneity", loc="left", fontweight="bold"); ax[0].set_xlabel("Publication year")
    for g_, c_ in (("cited_share_all89", "tab:blue"), ("cited_share_old33", "tab:red")): ax[1].plot(sh.index, sh[g_], "o-", ms=3, c=c_, label=g_)
    ax[1].set_title("(b) Cited share: 89 candidates vs old 33", loc="left", fontweight="bold"); ax[1].legend(fontsize=7, frameon=False)
    ax[2].plot(sh.index, sh.old33_share_of_cited, "o-", ms=3, label="old33 share of cited papers")
    ax[2].plot(sh.index, sh.old33_share_of_papers, "s--", ms=3, label="old33 share of all papers")
    ax[2].set_title("(c) Representativeness of the original sample", loc="left", fontweight="bold"); ax[2].legend(fontsize=7, frameon=False)
    plt.tight_layout(); plt.savefig(OUT / "fig_X3_heterogeneity.png", dpi=300, bbox_inches="tight"); plt.savefig(OUT / "fig_X3_heterogeneity.pdf", bbox_inches="tight"); plt.show()

    # ---------- 지도 1: 기간별 인용 비중(공통 색 범위) ----------
    g = xy(cand); first, last = pk[0], pk[-1]
    for pn, ys in PER.items():
        g[f"n_{pn}"] = pooled(N.loc[g.index], ys); g[f"s_{pn}"] = rate(ys)[[cand.index(i) for i in g.index]]
    vals = pd.concat([g.loc[g[f"n_{p}"] >= MIN_N_PERIOD, f"s_{p}"] for p in pk])
    nrm = Normalize(vals.quantile(.02), vals.quantile(.98)); cm = plt.cm.YlOrRd
    fig, axs = plt.subplots(1, len(pk), figsize=(6 * len(pk), 5.6)); axs = np.atleast_1d(axs)
    for ax_, pn in zip(axs, pk):
        gg = g[g[f"n_{pn}"] >= MIN_N_PERIOD]
        def draw(a, d, pn=pn):
            a.scatter(d.px, d.py, s=20 + 240 * np.sqrt(d[f"n_{pn}"] / g[[f"n_{p}" for p in pk]].max().max()), c=d[f"s_{pn}"], cmap=cm, norm=nrm,
                      edgecolor="k", linewidth=.4, alpha=.9, zorder=4)
        base_map(ax_, g); draw(ax_, gg); beijing_inset(ax_, gg, draw)
        ax_.set_title(f"{pn}: share of papers with ≥1 citing patent (n={len(gg)} institutions)", fontsize=9)
    fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cm), ax=axs, shrink=.6, label="cited share")
    plt.savefig(OUT / "map1_periods.png", dpi=300, bbox_inches="tight"); plt.savefig(OUT / "map1_periods.pdf", bbox_inches="tight"); plt.show()

    # ---------- 지도 2: 변화(마지막-첫 기간) 와 지도 3: 옛 33개 선정 여부 ----------
    both = g[(g[f"n_{first}"] >= MIN_N_PERIOD) & (g[f"n_{last}"] >= MIN_N_PERIOD)].copy()
    both["delta"] = both[f"s_{last}"] - both[f"s_{first}"]; vm = both.delta.abs().max()
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    def d2(a, d): a.scatter(d.px, d.py, s=60, c=d.delta, cmap="coolwarm", norm=TwoSlopeNorm(0, -vm, vm), edgecolor="k", linewidth=.4, zorder=4)
    base_map(axs[0], g); d2(axs[0], both); beijing_inset(axs[0], both, d2)
    axs[0].set_title(f"Change in cited share, {last} minus {first}", fontsize=9)
    fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, -vm, vm), cmap="coolwarm"), ax=axs[0], shrink=.6)
    g["old33"] = g.index.isin(OLD33); g["n_tot"] = N.loc[g.index].sum(axis=1).values
    def d3(a, d):
        for flag, c_, lb in ((True, "tab:red", "in original 33"), (False, "tab:blue", "added candidates")):
            s = d[d.old33 == flag]; a.scatter(s.px, s.py, s=20 + 240 * np.sqrt(s.n_tot / g.n_tot.max()), c=c_, edgecolor="k", linewidth=.4, alpha=.8, label=lb, zorder=4)
    base_map(axs[1], g); d3(axs[1], g); beijing_inset(axs[1], g, d3); axs[1].legend(loc="lower left", fontsize=7, frameon=False)
    axs[1].set_title("Original sample vs. candidate pool (size = papers)", fontsize=9)
    plt.savefig(OUT / "map2_3_change_and_sample.png", dpi=300, bbox_inches="tight"); plt.savefig(OUT / "map2_3_change_and_sample.pdf", bbox_inches="tight"); plt.show()

    # ---------- 선택: Pearson 잔차 Moran (같은 도시 링크 제외 포함) ----------
    def hav(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        a = np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la)[:, None] * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(a))
    def moran_perm(z, W):
        z = z - z.mean(); n = len(z); S0 = W.sum(); I = n / S0 * (z @ W @ z) / (z @ z)
        Z = np.array([RNG.permutation(z) for _ in range(4999)]); Ip = n / S0 * np.einsum("ri,ij,rj->r", Z, W, Z) / (z @ z)
        return I, (1 + np.sum(np.abs(Ip - Ip.mean()) >= abs(I - Ip.mean()))) / 5000
    rows = []
    for pn, ys in PER.items():
        c_, n_ = pooled(C, ys), pooled(N, ys)
        ok = (n_ >= MIN_N_PERIOD) & geo.lat.notna().values
        if ok.sum() < 12: continue
        pb = c_[ok].sum() / n_[ok].sum(); z = (c_[ok] / n_[ok] - pb) / np.sqrt(pb * (1 - pb) / n_[ok])
        D = hav(geo.lat.values[ok].astype(float), geo.lon.values[ok].astype(float))
        for wn, W in (("inv_dist(floor 25km)", 1 / np.maximum(D, 25)), ("inv_dist excl. same-city(<30km)", np.where(D < 30, 0, 1 / np.maximum(D, 25)))):
            W = W.copy(); np.fill_diagonal(W, 0); W = W / W.sum(1, keepdims=True).clip(min=1e-12)
            I, p = moran_perm(z, W); rows.append(dict(period=pn, weights=wn, n=int(ok.sum()), moran_I=I, p=p))
    if rows:
        RES["X3_5_moran"] = pd.DataFrame(rows); print("\n### X3-5 Moran (Pearson 잔차; 좌표 보유 기관만, 탐색적)"); print(RES["X3_5_moran"].round(4).to_string(index=False))
    else:
        print("\n[Moran 생략] 좌표 보유·표본 충족 기관 < 12개")

# ========================== 7. 패널이 없을 때의 대체 ===================
else:
    print("\n[대체 모드] 패널 파일 3개 연도 미만 → 옛 33개 CSV로 '인용된 논문 수'만 요약합니다 (비율 아님).")
    Cc = old[old.cit >= 1].groupby(["Institution", "year"]).size().unstack().fillna(0)
    RES["fallback_cited_counts"] = Cc; print(Cc.sum(axis=1).sort_values(ascending=False).head(15).to_string())
    g = xy(OLD33); g["cited_total"] = Cc.sum(axis=1).reindex(g.index).values
    fig, ax = plt.subplots(figsize=(8, 7)); base_map(ax, g)
    def dfb(a, d): a.scatter(d.px, d.py, s=20 + 300 * np.sqrt(d.cited_total / g.cited_total.max()), c="tab:red", edgecolor="k", alpha=.75, zorder=4)
    dfb(ax, g); beijing_inset(ax, g, dfb); ax.set_title("Cited papers per institution (original 33; counts, not rates)", fontsize=9)
    plt.savefig(OUT / "map_fallback_counts.png", dpi=300, bbox_inches="tight"); plt.show()
    print("→ panel_txt_CHECKLIST.csv에서 빠진 연도를 확인하고 파일을 추가한 뒤 이 셀을 다시 실행하세요.")

# ========================== 8. 저장 ==================================
for k, v in RES.items(): (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)).to_csv(OUT / f"{k}.csv", encoding="utf-8-sig")
print("\n완료. 저장 위치:", OUT)

# ---- cell 03 ----
#

# ---- cell 04 ----
# =====================================================================
# P0 | 새로 저장한 두 파일 구조 파악 (읽기 전용)
#  - patents/Transportation_Citing Patents.csv
#  - papers/Transportation_Works Cited by Patents.csv
# =====================================================================
import re, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 80); pd.set_option("display.max_colwidth", 70)

BASE = Path("~/Research/TransDiff-CN_R1").expanduser()
F_PAT = BASE / "lens_export" / "patents" / "Transportation_Citing Patents.csv"
F_PAP = BASE / "lens_export" / "papers"  / "Transportation_Works Cited by Patents.csv"
OLD   = BASE / "Transport_CN_Scholarly_Works.csv"
POP   = BASE / "data_R1" / "papers_unique.parquet"

def load(path):
    print("="*100); print(path.name)
    if not path.exists():
        print("  [없음] 경로 확인:", path); return None
    print(f"  크기: {path.stat().st_size/1e6:,.1f} MB")
    raw = path.read_bytes()[:200000]
    for enc in ("utf-8-sig", "utf-8", "cp949", "latin1"):
        try: raw.decode(enc); break
        except UnicodeDecodeError: continue
    print("  인코딩(추정):", enc)
    df = pd.read_csv(path, encoding=enc, encoding_errors="replace", low_memory=False, dtype=str)
    print(f"  행 {len(df):,} / 열 {df.shape[1]}")
    return df

def profile(df, name):
    if df is None: return
    df.columns = df.columns.str.strip()
    print(f"\n[{name}] 열 목록")
    rows = []
    for c in df.columns:
        s = df[c]; nn = s.notna().sum()
        ex = s.dropna().astype(str).head(2).str.slice(0, 60).tolist()
        num = pd.to_numeric(s, errors="coerce").notna().sum()
        rows.append(dict(col=c, non_null=nn, null_pct=round(100*(1-nn/len(df)), 1), unique=s.nunique(),
                         numeric_share=round(num/max(nn, 1), 2), example=ex))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n[{name}] 앞 3행 (일부 열)")
    print(df.iloc[:3, :min(10, df.shape[1])].to_string())
    print(f"\n[{name}] 완전 중복 행:", int(df.duplicated().sum()))

pat = load(F_PAT); pap = load(F_PAP)
profile(pat, "Citing Patents"); profile(pap, "Works Cited by Patents")

# ---- 날짜/연도 열 자동 탐색 ----
def date_scan(df, name):
    if df is None: return
    cands = [c for c in df.columns if re.search(r"date|year|filing|priority|publication|application", c, re.I)]
    print(f"\n[{name}] 날짜·연도 후보 열:", cands or "없음")
    for c in cands:
        d = pd.to_datetime(df[c], errors="coerce")
        y = d.dt.year if d.notna().mean() > 0.5 else pd.to_numeric(df[c], errors="coerce")
        if y.notna().sum() == 0: print(f"  {c}: 해석 불가"); continue
        print(f"  {c}: 해석률 {y.notna().mean():.1%}, 범위 {int(y.min())}~{int(y.max())}")
        print("   연도 분포(상위):", y.value_counts().sort_index().tail(8).to_dict())
date_scan(pat, "Citing Patents"); date_scan(pap, "Works Cited by Patents")

# ---- 두 파일을 연결할 열 탐색 ----
def link_scan():
    if pat is None or pap is None: return
    print("\n" + "="*100); print("연결 가능성 점검")
    print("특허 파일 열 중 논문과 연결될 수 있는 후보:",
          [c for c in pat.columns if re.search(r"cited|scholar|npl|lens id|work|doi|pmid|mag|microsoft", c, re.I)])
    print("논문 파일 열 중 특허와 연결될 수 있는 후보:",
          [c for c in pap.columns if re.search(r"patent|citing", c, re.I)])
    # 공통 열 이름
    print("두 파일의 공통 열 이름:", sorted(set(pat.columns) & set(pap.columns)) or "없음")
link_scan()

# ---- 논문 파일: 기존 모집단과의 대조 ----
def pop_check():
    if pap is None: return
    idc = next((c for c in pap.columns if c.lower() == "lens id"), None)
    print("\n" + "="*100); print("논문 파일 ↔ 기존 213k 모집단 대조")
    if idc is None: print("  'Lens ID' 열이 없습니다. 열 목록을 확인하세요."); return
    ids = set(pap[idc].dropna().astype(str))
    print(f"  고유 Lens ID: {len(ids):,}")
    if POP.exists():
        U = pd.read_parquet(POP, columns=["Lens ID", "Publication Year", "Citing Patents Count"])
        U["cit"] = pd.to_numeric(U["Citing Patents Count"], errors="coerce")
        uid = set(U["Lens ID"].astype(str)); cited = set(U.loc[U.cit >= 1, "Lens ID"].astype(str))
        print(f"  모집단 인용≥1 논문 {len(cited):,}편 / 이 파일과 겹침 {len(ids & cited):,} "
              f"(파일 중 모집단에 있는 비율 {len(ids & uid)/max(len(ids),1):.1%}, 모집단 인용 논문 중 포함 비율 {len(ids & cited)/max(len(cited),1):.1%})")
        if "Citing Patents Count" in pap.columns:
            a = pd.to_numeric(pap["Citing Patents Count"], errors="coerce"); print("  이 파일의 Citing Patents Count 합:", int(a.sum()), "| 최솟값:", a.min())
    if OLD.exists():
        o = pd.read_csv(OLD, encoding="utf-8-sig", encoding_errors="replace", dtype=str, low_memory=False)
        print(f"  옛 33개 파일의 Lens ID와 겹침: {len(ids & set(o['Lens ID'].dropna())):,}")
    ycol = next((c for c in pap.columns if c.lower() == "publication year"), None)
    if ycol: print("  연도별 편수:", pd.to_numeric(pap[ycol], errors='coerce').value_counts().sort_index().to_dict())
pop_check()

# ---- 특허 파일: 간단 점검 ----
def pat_check():
    if pat is None: return
    idc = next((c for c in pat.columns if re.search(r"lens id", c, re.I)), None)
    print("\n" + "="*100); print("특허 파일 점검")
    if idc: print(f"  특허 ID 열 '{idc}': 고유 {pat[idc].nunique():,} / 행 {len(pat):,}")
    for c in pat.columns:
        if re.search(r"jurisdiction|country|kind|type|applicant|owner", c, re.I):
            print(f"  [{c}] 상위값:", pat[c].value_counts().head(6).to_dict())
pat_check()
print("\n※ 위 출력 전체(특히 '열 목록' 표, 날짜 후보 열, 연결 가능성 점검)를 붙여 주세요.")

# ---- cell 05 ----
#

# ---- cell 06 ----
# =====================================================================
# TransDiff-CN R1 | v7: 특허 날짜 기반 논문별 고정 인용 창 + 노출 보정 + 위약 순열
# 입력 : patents/Transportation_Citing Patents.csv
#        papers/Transportation_Works Cited by Patents.csv
#        data_R1/papers_unique.parquet (213,328편 모집단)
# 필요 : pandas numpy scipy statsmodels matplotlib (pyarrow 권장)
# 모든 표/그림은 RES 한 곳에서 생성 -> 표·그림 불일치 방지
# =====================================================================
import warnings, re, json, hashlib, itertools, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 60)

# ----------------------------- 설정 ---------------------------------
BASE   = Path("~/Research/TransDiff-CN_R1").expanduser()
F_PAT  = BASE / "lens_export" / "patents" / "Transportation_Citing Patents.csv"
F_PAP  = BASE / "lens_export" / "papers"  / "Transportation_Works Cited by Patents.csv"
POP    = BASE / "data_R1" / "papers_unique.parquet"
POPGZ  = BASE / "data_R1" / "papers_unique.csv.gz"
OUT    = BASE / "outputs_R1_v7"; OUT.mkdir(exist_ok=True)

EVENT      = 2013
YMIN, YMAX = 2007, 2023
# 특허 날짜가 '완전하다'고 보는 마지막 연도(추출 2026-11, 특허는 출원 후 약 18개월 뒤 공개되므로 보수적으로 설정)
HZ = {"pri": 2023, "app": 2023, "pub": 2025}
HZ_SENS = {"pri": 2024, "app": 2024, "pub": 2026}      # 민감도용(느슨한 기준)
KS         = (3, 5)                                    # 인용 창(년)
MATURE     = (2007, 2010)                              # 노출 보정에 쓸 성숙 코호트
N_DRAW     = 300                                       # 위약 묶음 수 (본 실행 500~1000, 빠른 점검 100)
PREV_TOL, MIN_TERM_N, MIN_GROUP_N, MAX_TERM_SHARE = 0.15, 200, 300, 0.25
ALPHA, SEED = 0.05, 20261122
RNG = np.random.default_rng(SEED)

# 노출 사전(결과를 보기 전에 확정한 것, v3~v5와 동일; 응답서에 첨부)
BRI_KW = {"railway": r"railway", "high_speed_rail": r"high-speed rail", "rail_transit": r"rail transit",
          "port": r"\bports?\b", "shipping": r"shipping", "maritime": r"maritime", "container": r"container",
          "freight": r"freight", "logistics": r"logistics", "belt_road": r"belt and road",
          "supply_chain": r"supply chain", "intermodal": r"intermodal", "cross_border": r"cross-border",
          "corridor": r"corridor"}
AMBIG = {"container", "supply_chain", "cross_border", "corridor", "belt_road", "intermodal"}
NEV_RX = r"electric vehicle|battery|charging|new energy vehicle|hybrid vehicle|fuel cell"
PLC_RX = r"traffic signal|pavement|asphalt|driver|pedestrian|crash|road safety|traffic flow"
TR_BROAD = (r"transport|traffic|railway|\brail\b|vehicle|\broads?\b|highway|logistic|shipping|aviation|airport|"
            r"\bports?\b|freight|transit|pavement|automotive|driving|maritime|mobility|intermodal")
ACAD_RX = r"UNIV|UNIVERSITY|INST\b|INSTITUTE|ACAD|COLLEGE|SCHOOL"
TARGETS = ["BRI_full[txt]", "BRI_narrow[txt]", "BRI_full[fos]", "BRI_narrow[fos]", "NEV[fos]", "Placebo_roadsafety[fos]"]

RES = {}
def keep(name, obj, show=True, n=60):
    RES[name] = obj
    if show:
        print(f"\n### {name}")
        print(obj.head(n).to_string() if isinstance(obj, pd.DataFrame) else obj)

# =========================== 0. 로드 ================================
U = pd.read_parquet(POP) if POP.exists() else pd.read_csv(POPGZ, low_memory=False)
U.columns = U.columns.str.strip()
U["year"] = pd.to_numeric(U["Publication Year"], errors="coerce")
U["cit"]  = pd.to_numeric(U["Citing Patents Count"], errors="coerce")
U = U.dropna(subset=["year", "cit"]).reset_index(drop=True); U["year"] = U.year.astype(int)
U = U[U.year.between(YMIN, YMAX)].reset_index(drop=True)
N = len(U); YEARS = list(range(YMIN, YMAX + 1)); NY = len(YEARS); yidx = U.year.values - YMIN
ymap = dict(zip(U["Lens ID"].astype(str), U.year))
print(f"모집단 {N:,}편, 인용≥1 {int((U.cit>=1).sum()):,}편")

cols = ["Lens ID", "Jurisdiction", "Kind", "Publication Date", "Application Date", "Earliest Priority Date",
        "Applicants", "Document Type", "Simple Family Members", "Simple Family Member Jurisdictions",
        "NPL Resolved Lens ID(s)"]
P = pd.read_csv(F_PAT, encoding="utf-8-sig", usecols=lambda c: c.strip() in cols, dtype=str, low_memory=False)
P.columns = P.columns.str.strip()
for c, nm in (("Earliest Priority Date", "pri"), ("Application Date", "app"), ("Publication Date", "pub")):
    P[nm + "_y"] = pd.to_datetime(P[c], errors="coerce").dt.year
print(f"특허 {len(P):,}건, 날짜 결측 (pri/app/pub): {P.pri_y.isna().sum()}/{P.app_y.isna().sum()}/{P.pub_y.isna().sum()}")

# =========================== 1. 연결표 + 패밀리 중복 제거 ================
def famkey(mem, lid):
    m = [x for x in str(mem).split(";;") if x and x != "nan"]
    return min(m) if m else lid
P["fam"] = [famkey(m, l) for m, l in zip(P["Simple Family Members"], P["Lens ID"])]
def jset(js, j):
    s = {x for x in str(js).split(";;") if x and x != "nan"}
    return s or {str(j)}
P["jur_set"] = [jset(a, b) for a, b in zip(P["Simple Family Member Jurisdictions"], P["Jurisdiction"])]
P["cn_only"] = P.jur_set.map(lambda s: s <= {"CN"})
P["has_us"]  = P.jur_set.map(lambda s: "US" in s)
P["acad"]    = P["Applicants"].fillna("").str.upper().str.contains(ACAD_RX, regex=True)

P["npl"] = P["NPL Resolved Lens ID(s)"].fillna("").str.split(";;")
L = P[["Lens ID", "fam", "pri_y", "app_y", "pub_y", "cn_only", "has_us", "acad", "npl", "Jurisdiction"]].explode("npl")
L = L.rename(columns={"npl": "paper_id", "Lens ID": "pid"})
L = L[L.paper_id.fillna("") != ""]
n_links_raw = len(L); L = L[L.paper_id.isin(ymap)].copy()
print(f"NPL 해석 링크 {n_links_raw:,}개 중 모집단 논문 {len(L):,}개")
L["py"] = L.paper_id.map(ymap)

Fm = (L.groupby(["paper_id", "fam"]).agg(pri=("pri_y", "min"), app=("app_y", "min"), pub=("pub_y", "min"),
      cn_only=("cn_only", "all"), has_us=("has_us", "any"), acad=("acad", "any"), py=("py", "first")).reset_index())
print(f"논문-패밀리 연결 {len(Fm):,}개 (특허 문서 단위 {len(L):,}개 → 패밀리 단위로 {1-len(Fm)/len(L):.1%} 감소)")

# ---- QA: Lens의 Citing Patents Count와 대조 ----
lk_docs = L.groupby("paper_id").pid.nunique(); lk_fam = Fm.groupby("paper_id").fam.nunique()
Q = U[["Lens ID", "year", "cit"]].copy(); Q["Lens ID"] = Q["Lens ID"].astype(str)
Q["linked_docs"] = Q["Lens ID"].map(lk_docs).fillna(0); Q["linked_fam"] = Q["Lens ID"].map(lk_fam).fillna(0)
cited = Q[Q.cit >= 1]
qa = pd.Series({"cited_papers(Lens count>=1)": len(cited),
                "of which with >=1 resolved link": int((cited.linked_docs >= 1).sum()),
                "link coverage": float((cited.linked_docs >= 1).mean()),
                "sum Lens Citing Patents Count": int(cited.cit.sum()), "sum linked patent docs": int(cited.linked_docs.sum()),
                "exact match share": float((cited.linked_docs == cited.cit).mean()),
                "papers with link but Lens count 0": int(((Q.cit == 0) & (Q.linked_docs >= 1)).sum()),
                "corr(linked_docs, Lens count)": float(np.corrcoef(cited.linked_docs, cited.cit)[0, 1])})
keep("00_link_QA", qa.to_frame("v"))
cov = cited.groupby("year").apply(lambda d: pd.Series({"cited_papers": len(d), "link_coverage": (d.linked_docs >= 1).mean(),
                                                       "docs_per_lens_count": d.linked_docs.sum() / d.cit.sum()}))
keep("00_link_coverage_by_year", cov.round(3))
print("※ link_coverage가 1에 가깝고 연도별로 안정적이어야 고정 창 결과를 믿을 수 있음. 낮은 해는 한계로 보고")

# 날짜 완전성 진단(호라이즌 선택 근거)
dd = pd.DataFrame({k: Fm[k].value_counts().sort_index() for k in ("pri", "app", "pub")}).loc[2015:2026]
keep("01_family_links_by_event_year", dd)
print("※ 뒤쪽 연도에서 건수가 급감하는 지점이 '날짜가 불완전한' 구간. HZ 설정의 근거로 사용")

# =========================== 2. 결과변수 정의 ===========================
def make_outcome(kind, k, dcol, hz):
    H = hz[dcol]; f = Fm[(Fm[dcol] <= H) & ((Fm[dcol] - Fm.py) <= k)]
    if kind == "nonCN": f = f[~f.cn_only]
    cnt = f.groupby("paper_id").size()
    c = U["Lens ID"].astype(str).map(cnt).fillna(0).values
    return (c >= 1).astype(float), c, min(YMAX, H - k)

OUTC = {}
OUTC["cum_any"] = dict(c1=(U.cit.values >= 1).astype(float), cnt=U.cit.values, last=YMAX, label="Cumulative (original metric)")
for k in KS:
    c1, c, last = make_outcome("any", k, "pri", HZ); OUTC[f"any{k}_pri"] = dict(c1=c1, cnt=c, last=last, label=f"Cited within {k}y (priority date)")
c1, c, last = make_outcome("nonCN", 5, "pri", HZ); OUTC["nonCN5_pri"] = dict(c1=c1, cnt=c, last=last, label="Cited within 5y by non-CN-only families")
c1, c, last = make_outcome("any", 5, "pub", HZ); OUTC["any5_pub"] = dict(c1=c1, cnt=c, last=last, label="Cited within 5y (publication date)")
c1, c, last = make_outcome("any", 5, "app", HZ); OUTC["any5_app"] = dict(c1=c1, cnt=c, last=last, label="Cited within 5y (application date)")
c1, c, last = make_outcome("any", 5, "pri", HZ_SENS); OUTC["any5_pri_H2024"] = dict(c1=c1, cnt=c, last=last, label="Cited within 5y (priority, looser horizon)")

rows = []
for nm, o in OUTC.items():
    for y in YEARS:
        if y > o["last"]: continue
        m = yidx == (y - YMIN); c1m, cm = o["c1"][m], o["cnt"][m]; ck = cm[cm >= 1]
        rows.append(dict(outcome=nm, year=y, papers=int(m.sum()), cited_share=c1m.mean(), cited_n=int(c1m.sum()),
                         mean_cnt_all=cm.mean(), share_eq1_among_cited=(ck == 1).mean() if len(ck) else np.nan,
                         share_ge3_among_cited=(ck >= 3).mean() if len(ck) else np.nan,
                         sd_log_among_cited=np.log1p(ck).std(ddof=1) if len(ck) > 2 else np.nan))
COH = pd.DataFrame(rows); keep("02_cohort_series_by_outcome", COH.round(4), show=False)
piv = COH.pivot(index="year", columns="outcome", values="cited_share")
print("\n### 02 코호트별 인용 논문 비중: 원 지표(cum_any) vs 고정 창"); print(piv.round(4).to_string())
print("※ 원 지표의 최근 하락(2018년 이후)이 고정 창에서는 어떻게 되는지가 R2 절단 지적에 대한 직접 답")

# 집계 시리즈 구조변화(고정 창)
def chow_F_at(y, i0, min_seg=3):
    n = len(y)
    if i0 < min_seg or n - i0 < min_seg: return np.nan
    A = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
    rss = lambda M, b: float(((b - M @ np.linalg.lstsq(M, b, rcond=None)[0]) ** 2).sum())
    rr, ru = rss(A, y), rss(A[:i0], y[:i0]) + rss(A[i0:], y[i0:])
    return ((rr - ru) / 2) / (ru / (n - 4)) if ru > 0 else np.nan
rows = []
for nm in OUTC:
    s = piv[nm].dropna()
    if len(s) < 9: continue
    ys = list(s.index); n = len(ys); prof = {ys[i]: chow_F_at(s.values, i) for i in range(3, n - 2)}
    prof = {a: b for a, b in prof.items() if not np.isnan(b)}
    rows.append(dict(outcome=nm, n_years=n, last_year=ys[-1], argmax_year=max(prof, key=prof.get),
                     F_2013=prof.get(EVENT, np.nan), rank_2013=sorted(prof, key=lambda a: -prof[a]).index(EVENT) + 1 if EVENT in prof else np.nan,
                     n_candidates=len(prof)))
keep("03_aggregate_break_profile", pd.DataFrame(rows).round(3))

# =========================== 3. 노출 보정(성숙 코호트 누적 곡선) ========
H0 = HZ["pri"]; M_ = H0 - MATURE[1]
first = Fm[Fm.pri <= H0].groupby("paper_id").pri.min()
fl = pd.DataFrame({"py": first.index.map(ymap), "first": first.values}); fl["lag"] = (fl["first"] - fl.py).clip(lower=0)
mat = fl[fl.py.between(*MATURE) & (fl.lag <= M_)]
Fcdf = np.array([(mat.lag <= a).mean() for a in range(0, M_ + 1)])
accr = pd.DataFrame({f"cohort {a}-{b}": [(fl[fl.py.between(a, b)].lag <= x).mean() * 1.0 for x in range(0, 9)]
                     for a, b in ((2007, 2010), (2011, 2013), (2014, 2016))}); accr.index.name = "lag (years)"
keep("04_accrual_curves_first_citation", accr.round(3))
print("※ 같은 lag에서 누적 비율이 코호트 간 비슷해야 '성숙 코호트 곡선으로 보정'이 정당함")

cum_by_y = pd.Series({y: OUTC["cum_any"]["c1"][yidx == y - YMIN].mean() for y in YEARS})
hz_share = {}   # 패밀리 링크 기준 호라이즌 시점 인용 비중(Lens 누적과 정의가 다를 수 있어 따로 계산)
cH = Fm[Fm.pri <= H0].groupby("paper_id").size()
cH_all = U["Lens ID"].astype(str).map(cH).fillna(0).values >= 1
rows = []
for y in YEARS:
    sh = cH_all[yidx == y - YMIN].mean(); E = max(H0 - y, 0)
    F_E = Fcdf[min(E, M_)] if E <= M_ else 1.0
    rows.append(dict(year=y, exposure_years=E, share_cited_by_H=sh, F_exposure=F_E, adjusted_share=sh / F_E if F_E > 0 else np.nan))
ADJ = pd.DataFrame(rows).set_index("year"); keep("05_exposure_adjusted_share", ADJ.round(4))
print("※ 보정은 '인용 지연 분포가 코호트 간 같다'는 가정에 의존. 노출이 짧은 해(exposure<=2)의 값은 불안정")

# =========================== 4. 특허 구성(중국 단독, 대학 출원인) =======
rows = []
for k in KS:
    f = Fm[(Fm.pri <= HZ["pri"]) & ((Fm.pri - Fm.py) <= k) & Fm.py.between(YMIN, HZ["pri"] - k)]
    g = f.groupby("py").agg(links=("fam", "size"), share_CN_only=("cn_only", "mean"), share_US=("has_us", "mean"),
                            share_academic_applicant=("acad", "mean")).reset_index(); g["k"] = k; rows.append(g)
COMP = pd.concat(rows); keep("06_citing_family_composition", COMP.round(3), show=False)
print("\n### 06 인용 패밀리의 구성(5년 창)"); print(COMP[COMP.k == 5].set_index("py").round(3).to_string())
print("※ CN 단독 패밀리와 대학 출원인 비중이 급증하면, 인용 증가가 '중국 특허 붐'이지 지식이전 개선이라고 말하기 어려움")

# =========================== 5. 위약 순열(고정 창 결과변수) =============
col = lambda nm: U[nm].fillna("").astype(str).str.lower() if nm in U.columns else pd.Series("", index=U.index)
fos = col("Fields of Study"); txt = col("Title") + " ; " + col("Keywords") + " ; " + fos
tpp = fos.str.split(r"\s*;\s*", regex=True)
Lm = pd.DataFrame({"row": np.repeat(np.arange(N), tpp.map(len).values), "term": list(itertools.chain.from_iterable(tpp.values))})
Lm["term"] = Lm.term.str.strip(); Lm = Lm[Lm.term != ""]
codes, vocab = pd.factorize(Lm.term)
X = sparse.csr_matrix((np.ones(len(Lm), dtype=np.float32), (Lm.row.values, codes)), shape=(N, len(vocab))); X.data[:] = 1
terms_idx = lambda rx: np.where(np.asarray(vocab.str.contains(rx, regex=True)))[0]
def tmask(rx):
    ix = terms_idx(rx); Xc = X.tocsc()
    return (np.asarray(Xc[:, ix].sum(axis=1)).ravel() > 0) if len(ix) else np.zeros(N, bool)
rx_from = lambda ks: "|".join(BRI_KW[k] for k in ks)
GRX = {"BRI_full": rx_from(BRI_KW), "BRI_narrow": rx_from([k for k in BRI_KW if k not in AMBIG]), "NEV": NEV_RX, "Placebo_roadsafety": PLC_RX}
MASK = {}
for g, rx in GRX.items():
    MASK[f"{g}[txt]"] = txt.str.contains(rx, regex=True).values; MASK[f"{g}[fos]"] = tmask(rx)
dict_terms = np.zeros(len(vocab), bool)
for rx in list(GRX.values()): dict_terms[terms_idx(rx)] = True
POPM = {"all": np.ones(N, bool), "T_broad": tmask(TR_BROAD)}

PLC_OUT = ["cum_any", "any3_pri", "any5_pri", "nonCN5_pri", "any5_pub"]   # 위약 순열에 쓸 결과변수

def stats_lr(lr, yrs):
    yrs = np.array(yrs); i0 = int(np.where(yrs == EVENT)[0][0]); b = np.polyfit(yrs[:i0], lr[:i0], 1)
    k = yrs != 2012; pre, post = (yrs < EVENT) & k, (yrs >= EVENT) & k
    return dict(DID_long=lr[yrs >= EVENT].mean() - lr[:i0].mean(), DID_drop2012=lr[post].mean() - lr[pre].mean(),
                TA=(lr[i0:] - np.polyval(b, yrs[i0:])).mean(), ChowF=chow_F_at(lr, i0), pre_slope=b[0])

def lr_stats(m, pc, pn_all, pc_all, ytab, valid):
    ng = np.bincount(ytab[m], minlength=NY).astype(float); cg = np.bincount(ytab[m], weights=pc[m], minlength=NY)
    nc, cc = pn_all - ng, pc_all - cg
    lr = (np.log((cg + .5) / (ng + 1)) - np.log((cc + .5) / (nc + 1)))[valid]
    pre = np.array(YEARS)[valid] < EVENT
    return lr, cg[valid][pre].sum() / max(ng[valid][pre].sum(), 1)

def pv(nd, o, key):
    v = nd[key].values.astype(float); ok = ~np.isnan(v); vv = v[ok]; ov = o[key]
    if len(vv) < 30 or np.isnan(ov): return dict(p_emp=np.nan, p_adj=np.nan, null_med=np.nan, q025=np.nan, q975=np.nan)
    up = key == "ChowF"; med = np.median(vv)
    p_emp = (1 + np.sum(vv >= ov)) / (len(vv) + 1) if up else (1 + np.sum(np.abs(vv - med) >= abs(ov - med))) / (len(vv) + 1)
    A = np.column_stack([np.ones(len(nd)), nd.pre_share.values, nd.pre_slope.values])[ok]
    tf = (lambda a: np.log(np.maximum(a, 1e-9))) if up else (lambda a: a)
    beta = np.linalg.lstsq(A, tf(vv), rcond=None)[0]; rn = tf(vv) - A @ beta
    ro = tf(ov) - np.array([1, o["pre_share"], o["pre_slope"]]) @ beta
    p_adj = (1 + np.sum(rn >= ro)) / (len(rn) + 1) if up else (1 + np.sum(np.abs(rn - np.median(rn)) >= abs(ro - np.median(rn)))) / (len(rn) + 1)
    return dict(p_emp=p_emp, p_adj=p_adj, null_med=med, q025=np.percentile(vv, 2.5), q975=np.percentile(vv, 97.5))

SUMR, OBSLR, NULLLR = [], {}, {}
for pn, pm in POPM.items():
    pidx = np.where(pm)[0]; ytab = yidx[pidx]; npop = len(pidx)
    Xp = X[pidx].tocsc(); tn = np.diff(Xp.indptr)
    cand = np.where((tn >= MIN_TERM_N) & (tn / npop <= MAX_TERM_SHARE) & ~dict_terms)[0]
    pn_all = np.bincount(ytab, minlength=NY).astype(float)
    pcs = {o: OUTC[o]["c1"][pidx] for o in PLC_OUT}
    pc_all = {o: np.bincount(ytab, weights=pcs[o], minlength=NY) for o in PLC_OUT}
    valids = {o: np.array([y <= OUTC[o]["last"] for y in YEARS]) for o in PLC_OUT}
    yrs_v = {o: [y for y in YEARS if y <= OUTC[o]["last"]] for o in PLC_OUT}
    for lab in TARGETS:
        m0 = MASK[lab][pidx]
        if m0.sum() < MIN_GROUP_N or m0.mean() > .5: print(f"[생략] {pn}/{lab}"); continue
        obs = {}; 
        for o in PLC_OUT:
            lr, ps = lr_stats(m0, pcs[o], pn_all, pc_all[o], ytab, valids[o])
            obs[o] = dict(**stats_lr(lr, yrs_v[o]), pre_share=ps); OBSLR[(pn, lab, o)] = lr
        # 위약 묶음: 같은 마스크로 모든 결과변수를 동시에 계산
        target = m0.mean(); rec = {o: [] for o in PLC_OUT}; lrs = {o: [] for o in PLC_OUT}; tries = 0
        while len(rec[PLC_OUT[0]]) < N_DRAW and tries < N_DRAW * 30:
            tries += 1; mask = np.zeros(npop, bool); cnt = 0; ok = False
            for t in RNG.permutation(cand):
                r = Xp.indices[Xp.indptr[t]:Xp.indptr[t + 1]]; add = int((~mask[r]).sum())
                if (cnt + add) / npop > target * (1 + PREV_TOL): continue
                mask[r] = True; cnt += add
                if cnt / npop >= target * (1 - PREV_TOL): ok = True; break
            if not ok: continue
            for o in PLC_OUT:
                lr, ps = lr_stats(mask, pcs[o], pn_all, pc_all[o], ytab, valids[o])
                rec[o].append(dict(**stats_lr(lr, yrs_v[o]), pre_share=ps)); lrs[o].append(lr)
        for o in PLC_OUT:
            nd = pd.DataFrame(rec[o]); NULLLR[(pn, lab, o)] = np.array(lrs[o])
            n_g = np.bincount(ytab[m0], minlength=NY); c_g = np.bincount(ytab[m0], weights=pcs[o][m0], minlength=NY)
            vy = valids[o]; pre = np.array(YEARS)[vy] < EVENT
            for key in ("DID_long", "DID_drop2012", "TA", "ChowF"):
                SUMR.append(dict(population=pn, target=lab, outcome=o, stat=key, observed=obs[o][key], n_null=len(nd),
                                 cited_pre=int(c_g[vy][pre].sum()), cited_post=int(c_g[vy][~pre].sum()), **pv(nd, obs[o], key)))
        print(f"[위약] {pn:8s} {lab:26s} 묶음 {int(m0.sum()):>6,}편, 유효 {len(rec[PLC_OUT[0]])}/{N_DRAW}")

SUM = pd.DataFrame(SUMR)
fam_ = SUM.target.str.startswith("BRI") & SUM.p_adj.notna()
SUM["bh_p_emp"] = np.nan; SUM["bh_p_adj"] = np.nan
for o in PLC_OUT:
    mk = fam_ & (SUM.outcome == o)
    if mk.any():
        SUM.loc[mk, "bh_p_emp"] = multipletests(SUM.loc[mk, "p_emp"], method="fdr_bh")[1]
        SUM.loc[mk, "bh_p_adj"] = multipletests(SUM.loc[mk, "p_adj"], method="fdr_bh")[1]
keep("07_placebo_full", SUM.round(4), show=False)
V = SUM.groupby(["outcome", "population", "target"]).apply(lambda d: pd.Series({
    "n_tests": len(d), "n_p_emp<.05": int((d.p_emp < ALPHA).sum()), "n_p_adj<.05": int((d.p_adj < ALPHA).sum()),
    "n_BH_adj<.05": int((d.bh_p_adj < ALPHA).sum()) if d.bh_p_adj.notna().any() else np.nan,
    "min_p_adj": d.p_adj.min(), "cited_pre": d.cited_pre.iloc[0], "cited_post": d.cited_post.iloc[0]}))
keep("08_verdict", V.round(4), n=200)
core = SUM[SUM.stat.isin(["DID_long", "DID_drop2012"])].pivot_table(
    index=["outcome", "population", "target"], columns="stat", values=["observed", "p_emp", "p_adj"])
keep("09_core_table_for_response", core.round(4), n=200)
print("※ 해석: 원 지표(cum_any)에서 신호가 있고 고정 창(any5_pri, any3_pri)에서 사라지면 절단 산물. "
      "고정 창에서도 남으면 BRI 연관 탐색 신호. nonCN5에서 사라지면 중국 특허 붐 의존. cited_pre가 수십 편이면 검정력 낮음")

# =========================== 6. 그림 ===================================
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
for nm, st in [("cum_any", "k-o"), ("any3_pri", "b-s"), ("any5_pri", "r-^"), ("nonCN5_pri", "g--v")]:
    s = piv[nm].dropna(); ax[0, 0].plot(s.index, s.values, st, ms=3, label=OUTC[nm]["label"])
ax[0, 0].axvline(EVENT, c="gray", ls=":", lw=1); ax[0, 0].set_title("(a) Share of papers cited by patents, by cohort", loc="left", fontweight="bold")
ax[0, 0].set_xlabel("Publication year"); ax[0, 0].legend(fontsize=6.5, frameon=False)

for c_ in accr.columns: ax[0, 1].plot(accr.index, accr[c_], "o-", ms=3, label=c_)
ax[0, 1].set_title("(b) Accrual of first patent citation (cumulative share)", loc="left", fontweight="bold")
ax[0, 1].set_xlabel("Years since publication"); ax[0, 1].legend(fontsize=7, frameon=False)

ax[0, 2].plot(ADJ.index, ADJ.share_cited_by_H, "o-", ms=3, label=f"Cited by {H0} (raw)")
ax[0, 2].plot(ADJ.index, ADJ.adjusted_share, "s--", ms=3, label="Exposure-adjusted")
ax[0, 2].set_title("(c) Exposure adjustment using mature cohorts", loc="left", fontweight="bold"); ax[0, 2].legend(fontsize=7, frameon=False)
ax[0, 2].set_xlabel("Publication year")

c5 = COMP[COMP.k == 5].set_index("py")
for cn, st in [("share_CN_only", "o-"), ("share_US", "s-"), ("share_academic_applicant", "^--")]: ax[1, 0].plot(c5.index, c5[cn], st, ms=3, label=cn)
ax[1, 0].set_title("(d) Composition of citing patent families (5y window)", loc="left", fontweight="bold"); ax[1, 0].legend(fontsize=7, frameon=False)
ax[1, 0].set_xlabel("Publication year of cited paper")

pn_, lab_ = ("T_broad", "BRI_full[fos]")
for j, o in enumerate(["cum_any", "any5_pri"]):
    key = (pn_, lab_, o)
    if key not in NULLLR: continue
    yy = [y for y in YEARS if y <= OUTC[o]["last"]]; nl = NULLLR[key]; q = np.percentile(nl, [5, 50, 95], axis=0)
    base0 = lambda a: a - a[:int(np.sum(np.array(yy) < EVENT))].mean()
    ax[1, 1 + j].fill_between(yy, base0(q[0]), base0(q[2]), alpha=.25, label="Random topic groups (5-95%)")
    ax[1, 1 + j].plot(yy, base0(OBSLR[key]), "o-", ms=3, label="BRI-related (observed)")
    ax[1, 1 + j].axvline(EVENT, c="k", ls=":", lw=1); ax[1, 1 + j].legend(fontsize=7, frameon=False)
    ax[1, 1 + j].set_title(f"({'ef'[j]}) Log ratio vs rest: {OUTC[o]['label']}", loc="left", fontweight="bold", fontsize=8)
    ax[1, 1 + j].set_xlabel("Publication year")
plt.tight_layout(); plt.savefig(OUT / "fig_R1_v7.png", dpi=300, bbox_inches="tight"); plt.savefig(OUT / "fig_R1_v7.pdf", bbox_inches="tight"); plt.show()

# =========================== 7. 저장 ===================================
for k, v in RES.items(): (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)).to_csv(OUT / f"{k}.csv", encoding="utf-8-sig")
Fm.to_csv(OUT / "paper_family_links.csv.gz", index=False, compression="gzip")
(OUT / "analysis_manifest_v7.json").write_text(json.dumps({
    "seed": SEED, "event": EVENT, "HZ": HZ, "HZ_SENS": HZ_SENS, "KS": KS, "mature": MATURE, "n_draw": N_DRAW,
    "bri_keywords": BRI_KW, "ambiguous": sorted(AMBIG), "nev_rx": NEV_RX, "placebo_rx": PLC_RX, "tr_broad": TR_BROAD,
    "acad_rx": ACAD_RX, "n_papers": int(N), "n_patents": int(len(P)),
    "md5_patents": hashlib.md5(F_PAT.read_bytes()).hexdigest(), "md5_papers": hashlib.md5(F_PAP.read_bytes()).hexdigest()},
    indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n완료. 표·그림·링크표·매니페스트: {OUT}")

# ---- cell 07 ----
#

# ---- cell 08 ----
# =====================================================================
# TransDiff-CN R1 | v8
#  (1) 고정 창 결과변수  (2) 고정 창 압축 진단 + 부트스트랩
#  (3) 위약 순열 1000회 + 본문용 표  (4) 그림  (5) R1 분류 검증  (6) manifest
# 입력 : data_R1/papers_unique.parquet, outputs_R1_v7/paper_family_links.csv.gz
# 필요 : pandas numpy scipy statsmodels matplotlib pyarrow
# =====================================================================
import warnings, json, hashlib, itertools
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 60)

# ----------------------------- 설정 ---------------------------------
BASE = Path("~/Research/TransDiff-CN_R1").expanduser()
POP  = BASE / "data_R1" / "papers_unique.parquet"; POPGZ = BASE / "data_R1" / "papers_unique.csv.gz"
FM   = BASE / "outputs_R1_v7" / "paper_family_links.csv.gz"
OUT  = BASE / "outputs_R1_v8"; OUT.mkdir(exist_ok=True)

EVENT, YMIN, YMAX = 2013, 2007, 2023
HZ_PRI = 2023                 # v7과 동일(우선일이 '완전한' 마지막 해)
N_DRAW = 1000                 # 빠른 점검은 100
B_BOOT = 1000
PREV_TOL, MIN_TERM_N, MIN_GROUP_N, MAX_TERM_SHARE = 0.15, 200, 300, 0.25
ALPHA, SEED = 0.05, 20261122
RNG = np.random.default_rng(SEED)
RUN_PLACEBO = True            # False면 블록 3 생략(블록 1·2·5만 빠르게)

# v3~v7에서 결과 확인 전에 확정한 사전(변경 금지)
BRI_KW = {"railway": r"railway", "high_speed_rail": r"high-speed rail", "rail_transit": r"rail transit",
          "port": r"\bports?\b", "shipping": r"shipping", "maritime": r"maritime", "container": r"container",
          "freight": r"freight", "logistics": r"logistics", "belt_road": r"belt and road",
          "supply_chain": r"supply chain", "intermodal": r"intermodal", "cross_border": r"cross-border",
          "corridor": r"corridor"}
AMBIG = {"container", "supply_chain", "cross_border", "corridor", "belt_road", "intermodal"}
NEV_RX = r"electric vehicle|battery|charging|new energy vehicle|hybrid vehicle|fuel cell"
PLC_RX = r"traffic signal|pavement|asphalt|driver|pedestrian|crash|road safety|traffic flow"
TR_BROAD = (r"transport|traffic|railway|\brail\b|vehicle|\broads?\b|highway|logistic|shipping|aviation|airport|"
            r"\bports?\b|freight|transit|pavement|automotive|driving|maritime|mobility|intermodal")
TARGETS = ["BRI_full[txt]", "BRI_narrow[txt]", "BRI_full[fos]", "BRI_narrow[fos]", "NEV[fos]", "Placebo_roadsafety[fos]"]
PRIMARY = "BRI_full[txt]"     # 사전 확정 주 사양

RES = {}
def keep(name, obj, show=True, n=60):
    RES[name] = obj
    if show:
        print(f"\n### {name}"); print(obj.head(n).to_string() if isinstance(obj, pd.DataFrame) else obj)

# =========================== 0. 로드 ================================
U = pd.read_parquet(POP) if POP.exists() else pd.read_csv(POPGZ, low_memory=False)
U.columns = U.columns.str.strip()
U["year"] = pd.to_numeric(U["Publication Year"], errors="coerce")
U["cit"]  = pd.to_numeric(U["Citing Patents Count"], errors="coerce")
U = U.dropna(subset=["year", "cit"]).reset_index(drop=True); U["year"] = U.year.astype(int)
U = U[U.year.between(YMIN, YMAX)].reset_index(drop=True)
N = len(U); YEARS = list(range(YMIN, YMAX + 1)); NY = len(YEARS); yidx = U.year.values - YMIN
ids = U["Lens ID"].astype(str)
Fm = pd.read_csv(FM)
print(f"모집단 {N:,}편 | 논문-패밀리 연결 {len(Fm):,}")

# =========================== 1. 고정 창 결과변수 ========================
def window_counts(k, H=HZ_PRI, nonCN=False, allow_neg=True):
    f = Fm[(Fm.pri <= H) & ((Fm.pri - Fm.py) <= k)]
    if not allow_neg: f = f[(f.pri - f.py) >= 0]
    if nonCN: f = f[~f.cn_only]
    return ids.map(f.groupby("paper_id").size()).fillna(0).values

OUTC = {"cum_any": dict(cnt=U.cit.values.astype(float), last=YMAX, label="Cumulative (original metric)")}
OUTC["any3_pri"]     = dict(cnt=window_counts(3), last=min(YMAX, HZ_PRI - 3), label="Cited within 3y of publication")
OUTC["any5_pri"]     = dict(cnt=window_counts(5), last=min(YMAX, HZ_PRI - 5), label="Cited within 5y of publication")
OUTC["any5_nonneg"]  = dict(cnt=window_counts(5, allow_neg=False), last=HZ_PRI - 5, label="5y, priority date not before publication")
OUTC["nonCN5_pri"]   = dict(cnt=window_counts(5, nonCN=True), last=HZ_PRI - 5, label="5y, excluding CN-only families")
for o in OUTC.values(): o["c1"] = (o["cnt"] >= 1).astype(float)

# =========================== 2. 고정 창 압축 진단 =======================
def comp_row(c):
    ck = c[c >= 1]
    return dict(papers=len(c), cited_n=len(ck), cited_share=len(ck) / len(c),
                share_eq1=(ck == 1).mean() if len(ck) else np.nan,
                share_ge3=(ck >= 3).mean() if len(ck) else np.nan,
                sd_log=np.log1p(ck).std(ddof=1) if len(ck) > 2 else np.nan,
                mean_cnt_cited=ck.mean() if len(ck) else np.nan)
rows = []
for nm, o in OUTC.items():
    for y in YEARS:
        if y > o["last"]: continue
        rows.append(dict(outcome=nm, year=y, **comp_row(o["cnt"][yidx == y - YMIN])))
COH = pd.DataFrame(rows); keep("v8_01_cohort_compression", COH.round(4), show=False)
for stat in ("cited_share", "share_eq1", "sd_log"):
    print(f"\n### {stat} by cohort (닫힌 코호트만; 누적 지표와 같은 해를 나란히 비교)")
    print(COH.pivot(index="year", columns="outcome", values=stat).round(3).to_string())
keep("v8_02_cohort_n_cited", COH.pivot(index="year", columns="outcome", values="cited_n"), show=False)
print("\n※ 셀 크기(cited_n)가 작은 초기 코호트(2007~2009)는 share_eq1·sd_log이 불안정. 해석 때 함께 표기")

def contrast(c, early, late, B=B_BOOT):
    """초기 vs 후기 코호트: 인용 논문 단위 부트스트랩으로 (후기 - 초기) 차이의 95% 구간"""
    yy = U.year.values
    ce, cl = c[(yy >= early[0]) & (yy <= early[1])], c[(yy >= late[0]) & (yy <= late[1])]
    ce, cl = ce[ce >= 1], cl[cl >= 1]
    f = lambda x: np.array([(x == 1).mean(), np.log1p(x).std(ddof=1)])
    obs = f(cl) - f(ce); bs = np.empty((B, 2))
    for b in range(B): bs[b] = f(RNG.choice(cl, len(cl))) - f(RNG.choice(ce, len(ce)))
    q = np.percentile(bs, [2.5, 97.5], axis=0)
    return [dict(stat="share_eq1", diff=obs[0], lo=q[0, 0], hi=q[1, 0], n_early=len(ce), n_late=len(cl)),
            dict(stat="sd_log", diff=obs[1], lo=q[0, 1], hi=q[1, 1], n_early=len(ce), n_late=len(cl))]
rows = []
for nm, late in (("cum_any", (2013, 2018)), ("any5_pri", (2013, 2018)), ("any3_pri", (2013, 2020)),
                 ("any5_nonneg", (2013, 2018)), ("nonCN5_pri", (2013, 2018)), ("cum_any", (2013, 2023))):
    for r in contrast(OUTC[nm]["cnt"], (2007, 2012), late):
        rows.append(dict(outcome=nm, early="2007-12", late=f"{late[0]}-{late[1]}", **r))
CON = pd.DataFrame(rows)
keep("v8_03_compression_contrast", CON.round(3))
print("※ 해석 규칙(사전 확정): any5_pri에서 구간이 0을 포함하면 압축은 고정 창에서 확인되지 않음 → 누적 지표의 압축은 절단 산물. "
      "구간이 0을 배제하면 '고정 창에서도 분포 압축이 있다'로 한정 서술(기관 간 수렴이 아님).")

# =========================== 3. 위약 순열 (본 실행) =====================
def chow_F_at(y, i0, min_seg=3):
    n = len(y)
    if i0 < min_seg or n - i0 < min_seg: return np.nan
    A = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
    rss = lambda M, b: float(((b - M @ np.linalg.lstsq(M, b, rcond=None)[0]) ** 2).sum())
    rr, ru = rss(A, y), rss(A[:i0], y[:i0]) + rss(A[i0:], y[i0:])
    return ((rr - ru) / 2) / (ru / (n - 4)) if ru > 0 else np.nan

def stats_lr(lr, yrs):
    yrs = np.array(yrs); i0 = int(np.where(yrs == EVENT)[0][0]); b = np.polyfit(yrs[:i0], lr[:i0], 1)
    k = yrs != 2012; pre, post = (yrs < EVENT) & k, (yrs >= EVENT) & k
    return dict(DID_long=lr[yrs >= EVENT].mean() - lr[:i0].mean(), DID_drop2012=lr[post].mean() - lr[pre].mean(),
                TA=(lr[i0:] - np.polyval(b, yrs[i0:])).mean(), ChowF=chow_F_at(lr, i0), pre_slope=b[0])

def lr_stats(m, pc, pn_all, pc_all, ytab, valid):
    ng = np.bincount(ytab[m], minlength=NY).astype(float); cg = np.bincount(ytab[m], weights=pc[m], minlength=NY)
    nc, cc = pn_all - ng, pc_all - cg
    lr = (np.log((cg + .5) / (ng + 1)) - np.log((cc + .5) / (nc + 1)))[valid]
    pre = np.array(YEARS)[valid] < EVENT
    return lr, cg[valid][pre].sum() / max(ng[valid][pre].sum(), 1)

def pv(nd, o, key):
    v = nd[key].values.astype(float); ok = ~np.isnan(v); vv = v[ok]; ov = o[key]
    if len(vv) < 30 or np.isnan(ov): return dict(p_emp=np.nan, p_adj=np.nan, null_med=np.nan, q025=np.nan, q975=np.nan)
    up = key == "ChowF"; med = np.median(vv)
    p_emp = (1 + np.sum(vv >= ov)) / (len(vv) + 1) if up else (1 + np.sum(np.abs(vv - med) >= abs(ov - med))) / (len(vv) + 1)
    A = np.column_stack([np.ones(len(nd)), nd.pre_share.values, nd.pre_slope.values])[ok]
    tf = (lambda a: np.log(np.maximum(a, 1e-9))) if up else (lambda a: a)
    beta = np.linalg.lstsq(A, tf(vv), rcond=None)[0]; rn = tf(vv) - A @ beta
    ro = tf(ov) - np.array([1, o["pre_share"], o["pre_slope"]]) @ beta
    p_adj = (1 + np.sum(rn >= ro)) / (len(rn) + 1) if up else (1 + np.sum(np.abs(rn - np.median(rn)) >= abs(ro - np.median(rn)))) / (len(rn) + 1)
    return dict(p_emp=p_emp, p_adj=p_adj, null_med=med, q025=np.percentile(vv, 2.5), q975=np.percentile(vv, 97.5))

NULLLR, OBSLR = {}, {}
if RUN_PLACEBO:
    col = lambda nm: U[nm].fillna("").astype(str).str.lower() if nm in U.columns else pd.Series("", index=U.index)
    fos = col("Fields of Study"); txt = col("Title") + " ; " + col("Keywords") + " ; " + fos
    tpp = fos.str.split(r"\s*;\s*", regex=True)
    Lm = pd.DataFrame({"row": np.repeat(np.arange(N), tpp.map(len).values), "term": list(itertools.chain.from_iterable(tpp.values))})
    Lm["term"] = Lm.term.str.strip(); Lm = Lm[Lm.term != ""]
    codes, vocab = pd.factorize(Lm.term)
    X = sparse.csr_matrix((np.ones(len(Lm), dtype=np.float32), (Lm.row.values, codes)), shape=(N, len(vocab))); X.data[:] = 1
    terms_idx = lambda rx: np.where(np.asarray(vocab.str.contains(rx, regex=True)))[0]
    def tmask(rx):
        ix = terms_idx(rx); Xc = X.tocsc()
        return (np.asarray(Xc[:, ix].sum(axis=1)).ravel() > 0) if len(ix) else np.zeros(N, bool)
    rx_from = lambda ks: "|".join(BRI_KW[k] for k in ks)
    GRX = {"BRI_full": rx_from(BRI_KW), "BRI_narrow": rx_from([k for k in BRI_KW if k not in AMBIG]),
           "NEV": NEV_RX, "Placebo_roadsafety": PLC_RX}
    MASK = {}
    for g, rx in GRX.items():
        MASK[f"{g}[txt]"] = txt.str.contains(rx, regex=True).values; MASK[f"{g}[fos]"] = tmask(rx)
    dict_terms = np.zeros(len(vocab), bool)
    for rx in GRX.values(): dict_terms[terms_idx(rx)] = True
    POPM = {"all": np.ones(N, bool), "T_broad": tmask(TR_BROAD)}
    PLC_OUT = ["cum_any", "any3_pri", "any5_pri", "nonCN5_pri"]

    SUMR = []
    for pn, pm in POPM.items():
        pidx = np.where(pm)[0]; ytab = yidx[pidx]; npop = len(pidx)
        Xp = X[pidx].tocsc(); tn = np.diff(Xp.indptr)
        cand = np.where((tn >= MIN_TERM_N) & (tn / npop <= MAX_TERM_SHARE) & ~dict_terms)[0]
        pn_all = np.bincount(ytab, minlength=NY).astype(float)
        pcs = {o: OUTC[o]["c1"][pidx] for o in PLC_OUT}
        pc_all = {o: np.bincount(ytab, weights=pcs[o], minlength=NY) for o in PLC_OUT}
        valids = {o: np.array([y <= OUTC[o]["last"] for y in YEARS]) for o in PLC_OUT}
        yrs_v = {o: [y for y in YEARS if y <= OUTC[o]["last"]] for o in PLC_OUT}
        for lab in TARGETS:
            m0 = MASK[lab][pidx]
            if m0.sum() < MIN_GROUP_N or m0.mean() > .5: print(f"[생략] {pn}/{lab}"); continue
            obs = {}
            for o in PLC_OUT:
                lr, ps = lr_stats(m0, pcs[o], pn_all, pc_all[o], ytab, valids[o])
                obs[o] = dict(**stats_lr(lr, yrs_v[o]), pre_share=ps); OBSLR[(pn, lab, o)] = lr
            target = m0.mean(); rec = {o: [] for o in PLC_OUT}; lrs = {o: [] for o in PLC_OUT}; tries = 0
            while len(rec[PLC_OUT[0]]) < N_DRAW and tries < N_DRAW * 30:
                tries += 1; mask = np.zeros(npop, bool); cnt = 0; ok = False
                for t in RNG.permutation(cand):
                    r = Xp.indices[Xp.indptr[t]:Xp.indptr[t + 1]]; add = int((~mask[r]).sum())
                    if (cnt + add) / npop > target * (1 + PREV_TOL): continue
                    mask[r] = True; cnt += add
                    if cnt / npop >= target * (1 - PREV_TOL): ok = True; break
                if not ok: continue
                for o in PLC_OUT:
                    lr, ps = lr_stats(mask, pcs[o], pn_all, pc_all[o], ytab, valids[o])
                    rec[o].append(dict(**stats_lr(lr, yrs_v[o]), pre_share=ps)); lrs[o].append(lr)
            for o in PLC_OUT:
                nd = pd.DataFrame(rec[o]); NULLLR[(pn, lab, o)] = np.array(lrs[o])
                c_g = np.bincount(ytab[m0], weights=pcs[o][m0], minlength=NY); vy = valids[o]; pre = np.array(YEARS)[vy] < EVENT
                for key in ("DID_long", "DID_drop2012", "TA", "ChowF"):
                    SUMR.append(dict(population=pn, target=lab, outcome=o, stat=key, observed=obs[o][key], n_null=len(nd),
                                     cited_pre=int(c_g[vy][pre].sum()), cited_post=int(c_g[vy][~pre].sum()), **pv(nd, obs[o], key)))
            print(f"[위약] {pn:8s} {lab:26s} 묶음 {int(m0.sum()):>6,}편, 유효 {len(rec[PLC_OUT[0]])}/{N_DRAW}")

    SUM = pd.DataFrame(SUMR)
    SUM["bh_p_emp"] = np.nan; SUM["bh_p_adj"] = np.nan
    fam_ = SUM.target.str.startswith("BRI") & SUM.p_adj.notna()
    for o in PLC_OUT:
        mk = fam_ & (SUM.outcome == o)
        if mk.any():
            SUM.loc[mk, "bh_p_emp"] = multipletests(SUM.loc[mk, "p_emp"], method="fdr_bh")[1]
            SUM.loc[mk, "bh_p_adj"] = multipletests(SUM.loc[mk, "p_adj"], method="fdr_bh")[1]
    keep("v8_04_placebo_full", SUM.round(4), show=False)

    # ---- 본문용 표: 전체 / 2012 제외, 인용 논문 수, 귀무 범위 ----
    W = SUM[SUM.stat.isin(["DID_long", "DID_drop2012"])]
    main = W.pivot_table(index=["outcome", "population", "target"], columns="stat",
                         values=["observed", "p_emp", "p_adj", "q025", "q975"]).round(3)
    main.columns = [f"{a}_{b}" for a, b in main.columns]   # 먼저 열을 1단계로 평탄화
    cn = W.groupby(["outcome", "population", "target"])[["cited_pre", "cited_post"]].first()
    main = main.join(cn)                                   # 이제 1단계끼리 결합
    keep("v8_05_main_table_for_manuscript", main, n=300)
    prim = main.xs(PRIMARY, level="target"); keep("v8_06_primary_spec_only", prim, n=100)
    V = SUM.groupby(["outcome", "population", "target"]).apply(lambda d: pd.Series({
        "n_p_emp<.05": int((d.p_emp < ALPHA).sum()), "n_p_adj<.05": int((d.p_adj < ALPHA).sum()),
        "n_BH_adj<.05": int((d.bh_p_adj < ALPHA).sum()) if d.bh_p_adj.notna().any() else np.nan,
        "min_p_adj": d.p_adj.min(), "cited_pre": d.cited_pre.iloc[0], "cited_post": d.cited_post.iloc[0]}))
    keep("v8_07_verdict", V.round(4), n=200)
    print("※ q025~q975 = 무작위 주제 묶음에서 나오는 DID의 95% 범위. 관측값이 이 범위 안이면 '탐지 불가', "
          "범위 폭 자체가 '이 데이터로 배제할 수 없는 효과 크기'의 근거")

# =========================== 4. 그림 ===================================
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
pv_ = COH.pivot(index="year", columns="outcome", values="cited_share")
for nm, st in [("cum_any", "k-o"), ("any3_pri", "b-s"), ("any5_pri", "r-^")]:
    s = pv_[nm].dropna(); ax[0].plot(s.index, s.values, st, ms=3, label=OUTC[nm]["label"])
ax[0].axvline(EVENT, c="gray", ls=":"); ax[0].set_xlabel("Publication year (cohort)")
ax[0].set_title("(a) Share of papers cited by patents: cumulative vs fixed windows", loc="left", fontweight="bold", fontsize=8.5)
ax[0].legend(fontsize=7, frameon=False)
for j, (stat, ttl) in enumerate((("share_eq1", "(b) Share with exactly 1 citing patent | cited"), ("sd_log", "(c) Paper-level SD of log(1+count) | cited"))):
    pp = COH.pivot(index="year", columns="outcome", values=stat)
    for nm, st in [("cum_any", "k-o"), ("any5_pri", "r-^"), ("nonCN5_pri", "g--v")]:
        s = pp[nm].dropna(); ax[1 + j].plot(s.index, s.values, st, ms=3, label=OUTC[nm]["label"])
    ax[1 + j].axvline(EVENT, c="gray", ls=":"); ax[1 + j].set_title(ttl, loc="left", fontweight="bold", fontsize=8.5)
    ax[1 + j].set_xlabel("Publication year (cohort)"); ax[1 + j].legend(fontsize=6.5, frameon=False)
plt.tight_layout(); plt.savefig(OUT / "fig_v8_A_truncation_compression.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT / "fig_v8_A_truncation_compression.pdf", bbox_inches="tight"); plt.show()

if RUN_PLACEBO:
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for j, o in enumerate(["cum_any", "any3_pri", "any5_pri"]):
        key = ("all", PRIMARY, o)
        if key not in NULLLR: continue
        yy = [y for y in YEARS if y <= OUTC[o]["last"]]; nl = NULLLR[key]; q = np.percentile(nl, [5, 50, 95], axis=0)
        i0 = int(np.sum(np.array(yy) < EVENT)); b0 = lambda a: a - a[:i0].mean()
        ax[j].fill_between(yy, b0(q[0]), b0(q[2]), alpha=.25, label="Random topic groups (5-95%)")
        ax[j].plot(yy, b0(OBSLR[key]), "o-", ms=3, label="BRI-related (pre-specified)")
        ax[j].axvline(EVENT, c="k", ls=":"); ax[j].legend(fontsize=7, frameon=False); ax[j].set_xlabel("Publication year")
        ax[j].set_title(f"({'abc'[j]}) Log ratio vs rest, {OUTC[o]['label']}", loc="left", fontweight="bold", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "fig_v8_B_placebo_primary.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / "fig_v8_B_placebo_primary.pdf", bbox_inches="tight"); plt.show()

# =========================== 5. R1: transport-research 분류 검증 ========
col = lambda nm: U[nm].fillna("").astype(str).str.lower() if nm in U.columns else pd.Series("", index=U.index)
fos_ = col("Fields of Study"); txt_ = col("Title") + " ; " + col("Keywords") + " ; " + fos_
tag = fos_.str.contains(TR_BROAD, regex=True).values
sh = pd.DataFrame({"transport_tag_share_all": pd.Series(tag).groupby(U.year.values).mean(),
                   "transport_tag_share_cited": pd.Series(tag[U.cit.values >= 1]).groupby(U.year.values[U.cit.values >= 1]).mean(),
                   "n_papers": U.groupby("year").size()})
keep("v8_08_transport_tag_share_by_year", sh.round(3))

smp_f = OUT / "validation_sample_200.csv"
if not smp_f.exists():
    per = {"2007-12": (2007, 2012), "2013-18": (2013, 2018), "2019-23": (2019, 2023)}; parts = []
    for lb, (a, b) in per.items():
        for t, n in ((True, 33), (False, 34)):          # 태그된 것/안 된 것을 구간마다 섞어 재현율도 점검
            ix = np.where((U.year.values >= a) & (U.year.values <= b) & (tag == t))[0]
            take = RNG.choice(ix, min(n, len(ix)), replace=False)
            parts.append(U.iloc[take][["Lens ID", "Title", "Fields of Study", "Keywords", "year"]].assign(regex_tag=t, stratum=lb))
    smp = pd.concat(parts).sample(frac=1, random_state=SEED)
    smp["rater1_is_transport"] = ""; smp["rater2_is_transport"] = ""
    smp.to_csv(smp_f, index=False, encoding="utf-8-sig")
    print(f"\n[검증 샘플] {smp_f} 생성 ({len(smp)}편). rater1/rater2 열에 1(교통 연구) 또는 0을 기입 후 이 셀을 다시 실행하세요.")
else:
    s = pd.read_csv(smp_f, encoding="utf-8-sig")
    s = s[s.rater1_is_transport.notna() & (s.rater1_is_transport.astype(str) != "")].copy()
    if len(s):
        r1 = s.rater1_is_transport.astype(int); tg = s.regex_tag.astype(bool)
        tp, fp, fn = int((tg & (r1 == 1)).sum()), int((tg & (r1 == 0)).sum()), int((~tg & (r1 == 1)).sum())
        out = dict(n=len(s), precision=tp / max(tp + fp, 1), recall_in_sample=tp / max(tp + fn, 1),
                   true_transport_share_in_sample=float(r1.mean()))
        if "rater2_is_transport" in s and s.rater2_is_transport.notna().all() and (s.rater2_is_transport.astype(str) != "").all():
            r2 = s.rater2_is_transport.astype(int); po = (r1 == r2).mean(); pe = r1.mean() * r2.mean() + (1 - r1.mean()) * (1 - r2.mean())
            out["cohen_kappa"] = (po - pe) / (1 - pe) if pe < 1 else np.nan
        keep("v8_09_classification_validation", pd.Series(out).to_frame("v").round(3))
        print("※ 샘플이 태그 여부로 층화되어 있어 precision은 직접, 전체 모집단의 재현율은 층 가중치로 환산해야 함(샘플 내 recall은 참고용)")

# =========================== 6. 저장 + manifest ========================
for k, v in RES.items(): (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)).to_csv(OUT / f"{k}.csv", encoding="utf-8-sig")
(OUT / "analysis_manifest_v8.json").write_text(json.dumps({
    "seed": SEED, "event": EVENT, "HZ_PRI": HZ_PRI, "N_DRAW": N_DRAW, "B_BOOT": B_BOOT, "primary": PRIMARY,
    "bri_keywords": BRI_KW, "ambiguous": sorted(AMBIG), "nev_rx": NEV_RX, "placebo_rx": PLC_RX, "tr_broad": TR_BROAD,
    "n_papers": int(N), "md5_family_links": hashlib.md5(FM.read_bytes()).hexdigest(),
    "note": "all tables/figures generated in this single script from v7 family-link file and population file"},
    indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n완료. 저장 위치: {OUT}")

# ---- cell 09 ----
#

# ---- cell 10 ----

