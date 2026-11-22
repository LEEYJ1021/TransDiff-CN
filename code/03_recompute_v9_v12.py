# ===== 03_recompute_v9_v12.py : generated from a notebook; code cells only, outputs removed =====

# ---- cell 00 ----
#

# ---- cell 01 ----
# =====================================================================
# TransDiff-CN R1 | v9: 확정본 교정용 단일 스크립트
# 입력 : data_R1/papers_unique.parquet, outputs_R1_v7/paper_family_links.csv.gz,
#        outputs_R1_v8/v8_04_placebo_full.csv, Transport_CN_Scholarly_Works.csv, gadm41_CHN_1.json
# 필요 : pandas numpy scipy matplotlib pyarrow (지도: geopandas)
# =====================================================================
import warnings, json, hashlib, itertools, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import sparse, stats
warnings.filterwarnings("ignore")
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 60)

# ----------------------------- 설정 ---------------------------------
BASE = Path("~/Research/TransDiff-CN_R1").expanduser()
POP, POPGZ = BASE/"data_R1/papers_unique.parquet", BASE/"data_R1/papers_unique.csv.gz"
FM   = BASE/"outputs_R1_v7/paper_family_links.csv.gz"
PLC  = BASE/"outputs_R1_v8/v8_04_placebo_full.csv"
OLD  = BASE/"Transport_CN_Scholarly_Works.csv"
GEOJSON = BASE/"gadm41_CHN_1.json"
OUT  = BASE/"outputs_R1_v9"; OUT.mkdir(exist_ok=True)

EVENT, YMIN, YMAX, HZ = 2013, 2007, 2023, 2023
N_DRAW, B_BOOT, N_PERM, N_PERM_MORAN = 1000, 1000, 500, 4999   # 빠른 점검: 100, 200, 100, 999
SEED = 20261122; RNG = np.random.default_rng(SEED)
RUN_PLACEBO, RUN_SPATIAL, RUN_MAPS = True, True, True
PRIMARY = "BRI_full[txt]"
YEARS = list(range(YMIN, YMAX + 1)); NY = len(YEARS)
OI = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "grey": "#666666"}   # 색각 이상 대응

# v3~v8에서 결과 확인 전에 확정한 사전(변경 금지)
BRI_KW = {"railway": r"railway", "high_speed_rail": r"high-speed rail", "rail_transit": r"rail transit",
          "port": r"\bports?\b", "shipping": r"shipping", "maritime": r"maritime", "container": r"container",
          "freight": r"freight", "logistics": r"logistics", "belt_road": r"belt and road",
          "supply_chain": r"supply chain", "intermodal": r"intermodal", "cross_border": r"cross-border",
          "corridor": r"corridor"}
BRI_RX = "|".join(BRI_KW.values())
NEV_RX = r"electric vehicle|battery|charging|new energy vehicle|hybrid vehicle|fuel cell"
PLC_RX = r"traffic signal|pavement|asphalt|driver|pedestrian|crash|road safety|traffic flow"
TR_BROAD = (r"transport|traffic|railway|\brail\b|vehicle|\broads?\b|highway|logistic|shipping|aviation|airport|"
            r"\bports?\b|freight|transit|pavement|automotive|driving|maritime|mobility|intermodal")

RES, CHK = {}, []
def keep(name, obj, show=True, n=80):
    RES[name] = obj
    if show:
        print(f"\n### {name}"); print(obj.head(n).to_string() if isinstance(obj, pd.DataFrame) else obj)
def chk(item, ms, new, tol=0.006):
    ok = isinstance(new, (int, float, np.floating)) and not np.isnan(new) and abs(ms - new) <= tol
    CHK.append(dict(item=item, manuscript=ms, recomputed=(round(float(new), 4) if isinstance(new, (int, float, np.floating)) else new), match=bool(ok)))

# =========================== 0. 로드 ================================
U = pd.read_parquet(POP) if POP.exists() else pd.read_csv(POPGZ, low_memory=False)
U.columns = U.columns.str.strip()
U["year"] = pd.to_numeric(U["Publication Year"], errors="coerce"); U["cit"] = pd.to_numeric(U["Citing Patents Count"], errors="coerce")
U = U.dropna(subset=["year", "cit"]).reset_index(drop=True); U["year"] = U.year.astype(int)
U = U[U.year.between(YMIN, YMAX)].reset_index(drop=True)
N = len(U); yy = U.year.values; yidx = yy - YMIN; ids = U["Lens ID"].astype(str)
Fm = pd.read_csv(FM); Fm["cn_only"] = Fm["cn_only"].astype(str).str.lower().eq("true")
col = lambda nm: U[nm].fillna("").astype(str).str.lower() if nm in U.columns else pd.Series("", index=U.index)
fos = col("Fields of Study"); txt = col("Title") + " ; " + col("Keywords") + " ; " + fos
tag = fos.str.contains(TR_BROAD, regex=True).values
print(f"모집단 {N:,}편 | 패밀리 링크 {len(Fm):,}")

def window_counts(k, nonCN=False):
    f = Fm[(Fm.pri <= HZ) & ((Fm.pri - Fm.py) <= k)]
    if nonCN: f = f[~f.cn_only]
    return ids.map(f.groupby("paper_id").size()).fillna(0).values
CNT = {"cumulative": U.cit.values.astype(float), "5y": window_counts(5), "3y": window_counts(3), "5y_nonCN": window_counts(5, True)}
LAST = {"cumulative": YMAX, "5y": HZ - 5, "3y": HZ - 3, "5y_nonCN": HZ - 5}

# 옛 33개 기관 파일
old = pd.read_csv(OLD, encoding="utf-8-sig", encoding_errors="replace", low_memory=False); old.columns = old.columns.str.strip()
old = old[old["Lens ID"].notna() & (old["Lens ID"].astype(str) != "Lens ID")].copy()
old["year"] = pd.to_numeric(old["Publication Year"], errors="coerce"); old["cit"] = pd.to_numeric(old["Citing Patents Count"], errors="coerce")
old = old.dropna(subset=["year", "cit", "Institution"]); old["year"] = old.year.astype(int); old = old[old.year.between(YMIN, YMAX)]
old["Institution"] = old.Institution.astype(str).str.replace(r"\s*\(.*\)$", "", regex=True).str.strip()
rows_iw = old.drop_duplicates(["Lens ID", "Institution"]).copy()          # 기관-논문 행
cell = rows_iw.groupby(["Institution", "year"]).cit.agg(mean="mean", n="size").reset_index()
cell["y"] = np.log1p(cell["mean"]); cell["dev"] = cell.y - cell.groupby("year").y.transform("mean")
INST = sorted(rows_iw.Institution.unique())
print(f"옛 파일: 기관 {len(INST)}개 | 기관-논문 행 {len(rows_iw):,} | 고유 논문 {rows_iw['Lens ID'].nunique():,} | 기관-연도 셀 {len(cell)} (원고: 33 / 11,271 / 9,294 / 541)")
chk("institutions", 33, len(INST), 0); chk("institution-work rows", 11271, len(rows_iw), 0)
chk("unique works", 9294, rows_iw["Lens ID"].nunique(), 0); chk("non-empty cells", 541, len(cell), 0)

# =========================== A. σ·SD 정의 규명 =========================
PER = {"2007-12": (2007, 2012), "2013-19": (2013, 2019), "2020-23": (2020, 2023), "2021-23": (2021, 2023)}
sy = cell.groupby("year").y.std(ddof=1)
D1 = {p: sy.loc[a:b].mean() for p, (a, b) in PER.items()}                                                   # 연도별 σ의 기간 평균
D2 = {p: np.log1p(rows_iw[rows_iw.year.between(a, b)].groupby("Institution").cit.mean()).std(ddof=1) for p, (a, b) in PER.items()}  # 기간 합산 평균의 σ
D3 = {p: cell[cell.year.between(a, b)].groupby("Institution").dev.mean().std(ddof=1) for p, (a, b) in PER.items()}                 # 연도 편차의 기간 평균의 σ
A1 = pd.DataFrame({"D1_mean_of_yearly_sigma": D1, "D2_sigma_of_pooled_inst_means": D2, "D3_sigma_of_period_avg_deviation": D3})
keep("A_sigma_definitions", A1.round(3))
print("※ 본문 0.478(2007-12)→0.105(2021-23)은 D1 열, 지도 제목 0.22/0.16/0.09는 D2 또는 D3 열과 대조. 한 정의로 통일하고 캡션에 정의를 쓸 것")
chk("sigma 2007-12 (text, D1)", 0.478, D1["2007-12"]); chk("sigma 2021-23 (text, D1)", 0.105, D1["2021-23"])
chk("map sigma 2007-12 (D3)", 0.22, D3["2007-12"], 0.01); chk("map sigma 2013-19 (D3)", 0.16, D3["2013-19"], 0.01); chk("map sigma 2020-23 (D3)", 0.09, D3["2020-23"], 0.01)

uw = rows_iw.drop_duplicates("Lens ID"); uw = uw[uw.cit >= 1]; rw = rows_iw[rows_iw.cit >= 1]
psd = lambda d: d.groupby("year").cit.apply(lambda s: np.log1p(s).std(ddof=1))
PSD = pd.DataFrame({"unique_works": psd(uw), "institution_work_rows": psd(rw)})
keep("A_paper_level_SD_33", PSD.round(3))
chk("paper SD 2007 (Sec 4.2, unique works)", 0.675, PSD.unique_works[2007]); chk("paper SD 2023 (Sec 4.2, unique works)", 0.222, PSD.unique_works[2023])
chk("paper SD 2007 (Fig 2a, rows)", 0.650, PSD.institution_work_rows[2007]); chk("paper SD 2023 (Fig 2a, rows)", 0.225, PSD.institution_work_rows[2023])
PSD.to_csv(OUT/"A_paper_level_SD_33.csv"); A1.to_csv(OUT/"A_sigma_definitions.csv")

# =========================== B. Table 1: 표본·2012 민감도·태그 하위표본 ===
def contrast(c, early, late, sub=None, drop=(), B=B_BOOT):
    m = np.ones(len(c), bool) if sub is None else sub; dr = ~np.isin(yy, drop)
    ce = c[m & dr & (yy >= early[0]) & (yy <= early[1])]; cl = c[m & dr & (yy >= late[0]) & (yy <= late[1])]
    ce, cl = ce[ce >= 1], cl[cl >= 1]
    if len(ce) < 20 or len(cl) < 20: return None
    f = lambda x: np.array([(x == 1).mean(), np.log1p(x).std(ddof=1)])
    obs = f(cl) - f(ce); bs = np.array([f(RNG.choice(cl, len(cl))) - f(RNG.choice(ce, len(ce))) for _ in range(B)])
    q = np.percentile(bs, [2.5, 97.5], axis=0)
    return dict(d_eq1=obs[0], eq1_lo=q[0, 0], eq1_hi=q[1, 0], d_sd=obs[1], sd_lo=q[0, 1], sd_hi=q[1, 1], n_early=len(ce), n_late=len(cl))
SAMP = {"pool_all": None, "pool_tagged": tag, "pool_nontagged": ~tag, "old33_works": ids.isin(set(old["Lens ID"])).values}
EARLY = {"2007-12": ((2007, 2012), ()), "2007-12 excl 2012": ((2007, 2012), (2012,)), "2007-11": ((2007, 2011), ())}
rows = []
for sm, sub in SAMP.items():
    for mt, c in CNT.items():
        for en, (e, dr) in EARLY.items():
            r = contrast(c, e, (2013, 2018), sub, dr)
            if r: rows.append(dict(sample=sm, metric=mt, early=en, late="2013-18", **r))
B1 = pd.DataFrame(rows); keep("B_table1_contrast", B1.round(3), n=200)
print("※ 해석 규칙: 'pool_all'이 원고 Table 1(표본=전체 후보 풀). 2012 제외·2007-11에서도 구간이 0을 배제하면 압축 결과는 2012 급등에 의존하지 않음. "
      "pool_tagged/nontagged가 같은 방향이면 구성 변화(태그 비중 하락)로는 설명되지 않음")
g = lambda mt, en="2007-12", sm="pool_all": B1[(B1["sample"] == sm) & (B1.metric == mt) & (B1.early == en)].iloc[0]
chk("Table1 cumulative dSD", -0.135, g("cumulative").d_sd); chk("Table1 5y dSD", -0.107, g("5y").d_sd)
chk("Table1 5y d_eq1", 0.041, g("5y").d_eq1); chk("Table1 5y nonCN dSD", -0.089, g("5y_nonCN").d_sd)
B1.to_csv(OUT/"B_table1_contrast.csv", index=False)

# =========================== C. 같은 노출 비교 (절단을 추론이 아닌 증거로) ==
fl = Fm.groupby("paper_id").agg(pri=("pri", "min"), py=("py", "first")); fl["lag"] = fl.pri - fl.py
ny = pd.Series(yy).value_counts(); LAGS = (0, 1, 2, 3, 5); rows = []
for y in YEARS:
    r = {"cohort": y}
    for e in LAGS:
        r[f"first_cite_within_{e}y"] = ((fl.py == y) & (fl.lag <= e)).sum() / ny[y] if y + e <= HZ else np.nan
    rows.append(r)
C1 = pd.DataFrame(rows).set_index("cohort"); keep("C_same_exposure", C1.round(4))
cmp_ = []
for e, late in ((0, [2021, 2022, 2023]), (1, [2021, 2022]), (2, [2021])):
    k = f"first_cite_within_{e}y"; cmp_.append(dict(lag=e, ref_2016_18=C1.loc[2016:2018, k].mean(), late_cohorts=f"{late[0]}-{late[-1]}", late=C1.loc[late, k].mean()))
C2 = pd.DataFrame(cmp_); C2["ratio"] = C2.late / C2.ref_2016_18; keep("C_same_exposure_summary", C2.round(4))
print("※ ratio≈1이면 같은 노출에서 2021-23 코호트의 인용 확률이 2016-18과 같다 → 누적 지표의 하락은 노출 부족. "
      "ratio<1이면 노출만으로 설명되지 않는 하락이 있으므로 'largely truncation'을 약화해야 함")
C1.to_csv(OUT/"C_same_exposure.csv"); C2.to_csv(OUT/"C_same_exposure_summary.csv", index=False)

# =========================== D. 위약 순열: 민감도·가짜 사건·p 병기 =========
SPECS = {"main(E=2013)": dict(E=2013), "drop2012": dict(E=2013, drop=(2012,)), "drop2013": dict(E=2013, drop=(2013,)),
         "drop2012&2013": dict(E=2013, drop=(2012, 2013)), "start2009": dict(E=2013, start=2009), "start2010": dict(E=2013, start=2010),
         "end2018": dict(E=2013, end=2018), "end2020": dict(E=2013, end=2020)}
def did(lr, yrs, E, drop=(), start=YMIN, end=YMAX):
    yrs = np.array(yrs); k = (yrs >= start) & (yrs <= end) & ~np.isin(yrs, drop); pre, post = k & (yrs < E), k & (yrs >= E)
    return lr[post].mean() - lr[pre].mean() if pre.sum() >= 2 and post.sum() >= 2 else np.nan
if RUN_PLACEBO:
    import itertools
    mask_p = txt.str.contains(BRI_RX, regex=True).values
    tpp = fos.str.split(r"\s*;\s*", regex=True)
    Lm = pd.DataFrame({"row": np.repeat(np.arange(N), tpp.map(len).values), "term": list(itertools.chain.from_iterable(tpp.values))})
    Lm["term"] = Lm.term.str.strip(); Lm = Lm[Lm.term != ""]; codes, vocab = pd.factorize(Lm.term)
    X = sparse.csr_matrix((np.ones(len(Lm), np.float32), (Lm.row.values, codes)), shape=(N, len(vocab))); X.data[:] = 1
    tidx = lambda rx: np.where(np.asarray(vocab.str.contains(rx, regex=True)))[0]
    def tmask(rx):
        ix = tidx(rx); return (np.asarray(X.tocsc()[:, ix].sum(axis=1)).ravel() > 0) if len(ix) else np.zeros(N, bool)
    dict_terms = np.zeros(len(vocab), bool)
    for rx in (BRI_RX, NEV_RX, PLC_RX): dict_terms[tidx(rx)] = True
    POPM = {"all": np.ones(N, bool), "T_broad": tmask(TR_BROAD)}
    OUTS = {"cumulative": "cumulative", "3y": "3y", "5y": "5y"}
    PREV_TOL, MIN_TERM_N, MAX_TERM_SHARE = 0.15, 200, 0.25
    def lr_series(m, pc, pn_all, pc_all, ytab, valid):
        ng = np.bincount(ytab[m], minlength=NY).astype(float); cg = np.bincount(ytab[m], weights=pc[m], minlength=NY)
        return (np.log((cg + .5) / (ng + 1)) - np.log((pc_all - cg + .5) / (pn_all - ng + 1)))[valid], ng[valid], cg[valid]
    NULL, OBS, GRP = {}, {}, []
    for pn, pm in POPM.items():
        pidx = np.where(pm)[0]; ytab = yidx[pidx]; npop = len(pidx); Xp = X[pidx].tocsc(); tn = np.diff(Xp.indptr)
        cand = np.where((tn >= MIN_TERM_N) & (tn / npop <= MAX_TERM_SHARE) & ~dict_terms)[0]
        pn_all = np.bincount(ytab, minlength=NY).astype(float)
        pcs = {o: (CNT[o][pidx] >= 1).astype(float) for o in OUTS}; pc_all = {o: np.bincount(ytab, weights=pcs[o], minlength=NY) for o in OUTS}
        valid = {o: np.array([y <= LAST[o] for y in YEARS]) for o in OUTS}; m0 = mask_p[pidx]; target = m0.mean()
        for o in OUTS:
            lr, ng, cg = lr_series(m0, pcs[o], pn_all, pc_all[o], ytab, valid[o]); OBS[(pn, o)] = lr
            GRP.append(pd.DataFrame(dict(population=pn, outcome=o, year=np.array(YEARS)[valid[o]], n_group=ng, cited_group=cg)))
        rec = {o: [] for o in OUTS}; tries = 0
        while len(rec["cumulative"]) < N_DRAW and tries < N_DRAW * 30:
            tries += 1; mask = np.zeros(npop, bool); cnt = 0; ok = False
            for t in RNG.permutation(cand):
                r = Xp.indices[Xp.indptr[t]:Xp.indptr[t + 1]]; add = int((~mask[r]).sum())
                if (cnt + add) / npop > target * (1 + PREV_TOL): continue
                mask[r] = True; cnt += add
                if cnt / npop >= target * (1 - PREV_TOL): ok = True; break
            if ok:
                for o in OUTS: rec[o].append(lr_series(mask, pcs[o], pn_all, pc_all[o], ytab, valid[o])[0])
        for o in OUTS: NULL[(pn, o)] = np.array(rec[o])
        print(f"[위약] {pn}: 무작위 묶음 {len(rec['cumulative'])}/{N_DRAW}")
    np.savez_compressed(OUT/"D_null_lr.npz", **{f"{a}__{b}": v for (a, b), v in NULL.items()})
    GR = pd.concat(GRP); keep("D_group_cells_primary", GR.round(0), show=False); GR.to_csv(OUT/"D_group_cells_primary.csv", index=False)
    print("\n### D 주 사양 묶음의 연도별 (논문 수 / 인용 논문 수) — 2012·2013 행 확인")
    print(GR[(GR.population == "all") & (GR.outcome == "5y")].set_index("year")[["n_group", "cited_group"]].T.to_string())

    rows = []
    for (pn, o), nl in NULL.items():
        yrs = [y for y in YEARS if y <= LAST[o]]; obs_lr = OBS[(pn, o)]; cg = GR[(GR.population == pn) & (GR.outcome == o)].set_index("year").cited_group
        for sn, sp in SPECS.items():
            ob = did(obs_lr, yrs, **sp); nd = np.array([did(l, yrs, **sp) for l in nl]); nd = nd[~np.isnan(nd)]
            if np.isnan(ob) or len(nd) < 30: continue
            med = np.median(nd); k = [y for y in yrs if sp.get("start", YMIN) <= y <= sp.get("end", YMAX) and y not in sp.get("drop", ())]
            rows.append(dict(population=pn, outcome=o, spec=sn, DID=ob, p_emp=(1 + np.sum(np.abs(nd - med) >= abs(ob - med))) / (len(nd) + 1),
                             q025=np.percentile(nd, 2.5), q975=np.percentile(nd, 97.5), cited_pre=int(cg.reindex([y for y in k if y < 2013]).sum()),
                             cited_post=int(cg.reindex([y for y in k if y >= 2013]).sum()), MDE_rel_prob=float(np.exp(np.percentile(nd, 97.5)))))
    D1t = pd.DataFrame(rows); keep("D_sensitivity", D1t.round(3), n=200); D1t.to_csv(OUT/"D_sensitivity.csv", index=False)
    print("※ MDE_rel_prob = 무작위 묶음 95% 상한의 지수. 이 값 미만의 상대 인용 확률 변화는 이 데이터로 구분 불가")
    rows = []
    for (pn, o), nl in NULL.items():
        yrs = [y for y in YEARS if y <= LAST[o]]
        for E in range(2010, 2017):
            ob = did(OBS[(pn, o)], yrs, E); nd = np.array([did(l, yrs, E) for l in nl]); nd = nd[~np.isnan(nd)]
            if np.isnan(ob) or len(nd) < 30: continue
            med, sd = np.median(nd), nd.std(ddof=1); rows.append(dict(population=pn, outcome=o, event=E, DID=ob, z_vs_null=(ob - med) / sd,
                         p_emp=(1 + np.sum(np.abs(nd - med) >= abs(ob - med))) / (len(nd) + 1)))
    SC = pd.DataFrame(rows); SC["rank_of_2013_by_abs_z"] = SC.groupby(["population", "outcome"]).z_vs_null.transform(lambda s: s.abs().rank(ascending=False))
    keep("D_event_year_scan", SC[SC.event == 2013].round(3)); SC.to_csv(OUT/"D_event_year_scan.csv", index=False)
    print("※ 가짜 사건 연도 스캔에서 2013의 |z| 순위가 1위가 아니면 'BRI 시점이 특별하다'고 말할 수 없음")

    # ---- p_emp · p_adj · BH 병기 (v8 결과 그대로, Table 2 최종본) ----
    OMAP = {"cum_any": "cumulative", "any3_pri": "3y", "any5_pri": "5y", "nonCN5_pri": "5y_nonCN"}   # v8 이름 -> v9 이름
    V8 = pd.read_csv(PLC); V8["outcome"] = V8.outcome.map(OMAP)
    T2 = V8[(V8.target == PRIMARY) & V8.stat.isin(["DID_long", "DID_drop2012"])]
    T2 = T2.pivot_table(index=["outcome", "population"], columns="stat", values=["observed", "p_emp", "p_adj", "q025", "q975"]).round(3)
    T2.columns = ["_".join(c) for c in T2.columns]
    T2 = T2.join(V8[(V8.target == PRIMARY) & (V8.stat == "DID_long")].set_index(["outcome", "population"])[["cited_pre", "cited_post", "bh_p_emp", "bh_p_adj"]])
    keep("D_table2_final", T2, n=50); T2.to_csv(OUT/"D_table2_final.csv")
    print("※ p_emp = 무작위 묶음 중 관측 이상으로 극단인 비율(양측, 귀무 중앙값 기준). p_adj = 귀무 DID를 사전 인용 비중·사전 추세로 회귀한 잔차로 같은 비교. "
          "범위(q025~q975)는 p_emp와 같은 비보정 분포이므로 두 값을 함께 써야 표와 범위가 모순되지 않음")
    fx = T2.loc[[i for i in T2.index if i[0] in ("3y", "5y", "5y_nonCN")]]
    print(f"※ 고정 창 사전 인용 논문 범위(주 사양): {int(fx.cited_pre.min())}–{int(fx.cited_pre.max())} (원고 '7–20'과 대조); "
          f"누적: {int(T2.loc['cumulative'].cited_pre.min())}–{int(T2.loc['cumulative'].cited_pre.max())} (원고 '25–47')")
    chk("primary fixed-window cited_pre min", 7, float(fx.cited_pre.min()), 0); chk("primary fixed-window cited_pre max", 20, float(fx.cited_pre.max()), 0)

    # ---- 그림 4 재작성 ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for j, (o, lb) in enumerate((("cumulative", "Cumulative (original metric)"), ("3y", "Cited within 3 years"), ("5y", "Cited within 5 years"))):
        yrs = [y for y in YEARS if y <= LAST[o]]; i0 = int(np.sum(np.array(yrs) < EVENT)); q = np.percentile(NULL[("all", o)], [5, 50, 95], axis=0)
        b0 = lambda a: a - a[:i0].mean()
        ax[j].fill_between(yrs, b0(q[0]), b0(q[2]), color=OI["blue"], alpha=.22, label="Random topic groups: pointwise 5–95%")
        ax[j].plot(yrs, b0(OBS[("all", o)]), "o-", c=OI["verm"], ms=3.5, label="BRI-related group")
        ax[j].axvline(EVENT, c="k", ls=":", lw=1); ax[j].text(EVENT + .15, ax[j].get_ylim()[1] * .92, "2013: reference only", fontsize=7)
        for y in (2012, 2013):
            if y in yrs: ax[j].annotate(str(y), (y, b0(OBS[("all", o)])[yrs.index(y)]), fontsize=7, xytext=(3, -9), textcoords="offset points")
        ax[j].set_title(f"({'abc'[j]}) {lb}", loc="left", fontweight="bold", fontsize=9); ax[j].set_xlabel("Publication year"); ax[j].legend(fontsize=7, frameon=False, loc="upper left")
    ax[0].set_ylabel("Log ratio vs rest (pre-2013 mean = 0)")
    plt.tight_layout(); plt.savefig(OUT/"fig_v9_D_placebo_primary.png", dpi=300, bbox_inches="tight"); plt.close()

# =========================== E. σ 노이즈 기준선 + 2012 기여 =============
rows = []
for y in YEARS:
    d = rows_iw[rows_iw.year == y]; cd, _ = pd.factorize(d.Institution); c = d.cit.values.astype(float); cn = np.bincount(cd)
    sig = lambda v: np.log1p(np.bincount(cd, weights=v) / cn).std(ddof=1)
    nl = np.array([sig(RNG.permutation(c)) for _ in range(N_PERM)]); ob = sig(c)
    rows.append(dict(year=y, k_inst=len(cn), sigma_obs=ob, null_mean=nl.mean(), null_lo=np.percentile(nl, 2.5), null_hi=np.percentile(nl, 97.5),
                     p_upper=(1 + np.sum(nl >= ob)) / (N_PERM + 1), excess_var_share=max(0, (ob ** 2 - (nl ** 2).mean()) / ob ** 2)))
E1 = pd.DataFrame(rows).set_index("year"); keep("E_sigma_vs_noise", E1.round(3))
print("※ sigma_obs가 null 구간 안이면 그 해의 기관 간 산포는 표집잡음과 구분 불가. excess_var_share = 잡음 초과 분산 비중(0이면 기관 간 신호 없음)")
for p, (a, b) in PER.items(): print(f"  {p}: 평균 excess_var_share = {E1.loc[a:b, 'excess_var_share'].mean():.3f}, p_upper<.05 연도 수 = {int((E1.loc[a:b, 'p_upper'] < .05).sum())}/{b - a + 1}")
d12 = cell[cell.year == 2012].set_index("Institution"); base = d12.y.std(ddof=1)
loo = pd.DataFrame({"n_works": d12.n, "mean_cit": d12["mean"].round(2), "sigma_without": [d12.drop(i).y.std(ddof=1) for i in d12.index]}, index=d12.index)
loo["sigma_change"] = loo.sigma_without - base; keep("E_2012_leave_one_institution_out", loo.sort_values("sigma_change").head(8).round(3))
print(f"※ 2012 σ = {base:.3f}. 상위 기여 기관 제거 시 σ 변화가 크면 2012 급등은 소수 기관의 소수 논문 때문(R2·진단 근거)")
E1.to_csv(OUT/"E_sigma_vs_noise.csv"); loo.to_csv(OUT/"E_2012_loo.csv")
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.fill_between(E1.index, E1.null_lo, E1.null_hi, color=OI["blue"], alpha=.22, label="Label-permutation null (95%)")
ax.plot(E1.index, E1.sigma_obs, "o-", c=OI["verm"], ms=3.5, label="Observed σ of institution means"); ax.axvline(EVENT, c="k", ls=":", lw=1)
ax.set_xlabel("Publication year"); ax.set_ylabel("σ (log scale)"); ax.legend(fontsize=8, frameon=False); plt.tight_layout()
plt.savefig(OUT/"fig_v9_E_sigma_null.png", dpi=300, bbox_inches="tight"); plt.close()

# =========================== F. 공간: n=33, Table 3 정의, Fig 5·6 ========
K33 = {"Chinese Academy of Sciences": ("Beijing", 39.98, 116.33), "Tsinghua University": ("Beijing", 40.00, 116.33), "Peking University": ("Beijing", 39.99, 116.31),
 "Beihang University": ("Beijing", 39.98, 116.35), "Beijing Institute of Technology": ("Beijing", 39.96, 116.31), "Beijing Jiaotong University": ("Beijing", 39.95, 116.34),
 "University of Chinese Academy of Sciences": ("Beijing", 39.91, 116.24), "Beijing University of Technology": ("Beijing", 39.87, 116.48), "Tianjin University": ("Tianjin", 39.11, 117.17),
 "Harbin Institute of Technology": ("Harbin", 45.75, 126.68), "Jilin University": ("Changchun", 43.83, 125.29), "Dalian University of Technology": ("Dalian", 38.89, 121.54),
 "Shandong University": ("Jinan", 36.67, 117.07), "Xi'an Jiaotong University": ("Xi'an", 34.25, 108.98), "Sichuan University": ("Chengdu", 30.63, 104.08),
 "Southwest Jiaotong University": ("Chengdu", 30.70, 104.00), "Chongqing University": ("Chongqing", 29.57, 106.47), "Central South University": ("Changsha", 28.17, 112.93),
 "University of Science and Technology of China": ("Hefei", 31.84, 117.26), "Nanjing University": ("Nanjing", 32.06, 118.78), "Southeast University": ("Nanjing", 32.06, 118.80),
 "Tongji University": ("Shanghai", 31.28, 121.50), "Fudan University": ("Shanghai", 31.30, 121.50), "Shanghai Jiao Tong University": ("Shanghai", 31.03, 121.44),
 "Shanghai University": ("Shanghai", 31.32, 121.40), "Zhejiang University": ("Hangzhou", 30.30, 120.09), "Wuhan University": ("Wuhan", 30.54, 114.36),
 "Wuhan University of Technology": ("Wuhan", 30.51, 114.33), "Huazhong University of Science and Technology": ("Wuhan", 30.51, 114.41),
 "South China University of Technology": ("Guangzhou", 23.15, 113.34), "Sun Yat-sen University": ("Guangzhou", 23.10, 113.30), "Shenzhen University": ("Shenzhen", 22.53, 113.93),
 "Soochow University": ("Suzhou", 31.30, 120.63)}   # 근삿값: 제출 전 검증
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
K = {norm(k): v for k, v in K33.items()}; miss = [i for i in INST if norm(i) not in K]
if miss: print("\n[공간 생략] 좌표 매칭 실패:", miss); RUN_SPATIAL = False
if RUN_SPATIAL:
    geo = pd.DataFrame({"Institution": INST}); geo[["city", "lat", "lon"]] = [K[norm(i)] for i in INST]; geo = geo.set_index("Institution")
    def hav(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        a = np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la)[:, None] * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(a))
    DD = hav(geo.lat.values.astype(float), geo.lon.values.astype(float)); city = geo.city.values; same = city[:, None] == city[None, :]
    rs = lambda W: W / W.sum(1, keepdims=True).clip(min=1e-12)
    def Wm(D, kind, k=3):
        D = np.maximum(D, .5); n = len(D); np.fill_diagonal(D, np.inf)
        if kind == "nofloor": W = 1 / D
        elif kind == "floor25": W = 1 / np.maximum(D, 25)
        elif kind == "floor100": W = 1 / np.maximum(D, 100)
        elif kind == "excl_samecity": W = np.where(D < 30, 0, 1 / np.maximum(D, 25))
        else:
            W = np.zeros((n, n)); 
            for i in range(n): W[i, np.argsort(D[i])[:k]] = 1
        np.fill_diagonal(W, 0); return rs(W)
    def moran(z, W, nperm=N_PERM_MORAN):
        z = np.asarray(z, float); z = z - z.mean(); n = len(z); S0 = W.sum(); den = z @ z; I = n / S0 * (z @ W @ z) / den
        Z = np.array([RNG.permutation(z) for _ in range(nperm)]); Ip = n / S0 * np.einsum("ri,ij,rj->r", Z, W, Z) / den
        return I, (1 + np.sum(np.abs(Ip - Ip.mean()) >= abs(I - Ip.mean()))) / (nperm + 1), Ip
    def outcome(defn, a, b):
        if defn == "A_avg_year_deviation": s = cell[cell.year.between(a, b)].groupby("Institution").dev.mean()
        else: s = np.log1p(rows_iw[rows_iw.year.between(a, b)].groupby("Institution").cit.mean())
        return s.reindex(INST)
    MS_T3 = {"nofloor": (-0.324, .013), "floor25": (-0.143, .159), "floor100": (-0.076, .348), "excl_samecity": (-0.017, .727), "knn3": (-0.255, .064), "knn5": (-0.150, .160), "city_floor25": (-0.095, .646)}
    def run_kinds(z, ok):
        o = {}
        for kd in ("nofloor", "floor25", "floor100", "excl_samecity"): o[kd] = moran(z[ok], Wm(DD[np.ix_(ok, ok)].copy(), kd))[:2]
        o["knn3"] = moran(z[ok], Wm(DD[np.ix_(ok, ok)].copy(), "knn", 3))[:2]; o["knn5"] = moran(z[ok], Wm(DD[np.ix_(ok, ok)].copy(), "knn", 5))[:2]
        cdf = pd.DataFrame({"c": city[ok], "z": z[ok], "la": geo.lat.values[ok], "lo": geo.lon.values[ok]}).groupby("c").mean()
        o["city_floor25"] = moran(cdf.z.values, Wm(hav(cdf.la.values.astype(float), cdf.lo.values.astype(float)), "floor25"))[:2]; return o
    rows, best = [], None
    for dn in ("A_avg_year_deviation", "B_pooled_log1p_mean"):
        z = outcome(dn, 2013, 2019).values; ok = ~np.isnan(z); res = run_kinds(z, ok)
        dev = sum(abs(res[k][0] - MS_T3[k][0]) for k in MS_T3)
        for k, (I, p) in res.items(): rows.append(dict(definition=dn, n=int(ok.sum()), weights=k, I=I, p=p, manuscript_I=MS_T3[k][0], manuscript_p=MS_T3[k][1]))
        print(f"[Table 3 재구성] {dn}: n={int(ok.sum())}, |ΔI| 합 = {dev:.3f}"); best = (dn, dev) if best is None or dev < best[1] else best
    T3 = pd.DataFrame(rows); keep("F_table3_definition_check", T3.round(3), n=40); T3.to_csv(OUT/"F_table3_definition_check.csv", index=False)
    SEL = best[0]; print(f"→ 원고 Table 3과 가장 가까운 정의: {SEL}. §3.3에 이 정의를 쓰고, |ΔI|가 크면(>0.05) Table 3 값을 재계산 값으로 교체")
    z0 = outcome(SEL, 2013, 2019); ok0 = ~z0.isna().values
    chk("Table3 nofloor I", -0.324, T3[(T3.definition == SEL) & (T3.weights == "nofloor")].I.iloc[0], 0.01)
    chk("Table3 floor25 I", -0.143, T3[(T3.definition == SEL) & (T3.weights == "floor25")].I.iloc[0], 0.01)
    print(f"  same-city 가중치 비중(no floor): {Wm(DD.copy(), 'nofloor')[same].sum() / Wm(DD.copy(), 'nofloor').sum():.3f} (원고 0.556)")

    # 쌍 단위 검정(Fig 5c 제목과 본문 일치용)
    iu = np.triu_indices(33, 1); v0 = z0.fillna(z0.mean()).values
    gap = lambda v: np.abs(v[:, None] - v[None, :])[iu]; rho = stats.spearmanr(gap(v0), DD[iu])[0]
    rp = np.array([stats.spearmanr(gap(RNG.permutation(v0)), DD[iu])[0] for _ in range(N_PERM_MORAN)]); p_rho = (1 + np.sum(np.abs(rp) >= abs(rho))) / (len(rp) + 1)
    print(f"[Fig 5c] ρ = {rho:.3f}, 순열 p = {p_rho:.3f}, n = {len(v0)} (원고: ρ=−0.03, p=.72)"); chk("Spearman rho", -0.03, rho, 0.01); chk("Spearman p", 0.72, p_rho, 0.03)

    # 표집잡음 보정(Pearson z) + 단독 논문 + 공저 가중치
    a, b = 2013, 2019; wk = rows_iw[rows_iw.year.between(a, b)].copy(); wk["n_inst"] = wk.groupby("Lens ID").Institution.transform("nunique")
    ex = wk[wk.n_inst == 1].assign(l=lambda d: np.log1p(d.cit)); s_w = ex.l.std(ddof=1)
    g_ = ex.groupby("Institution").l.agg(["mean", "size"]).reindex(INST); okx = (g_["size"] >= 5).values
    zz = ((g_["mean"] - ex.l.mean()) / (s_w / np.sqrt(g_["size"]))).values
    co = pd.DataFrame(0.0, index=INST, columns=INST)
    for _, d in wk[wk.n_inst > 1].groupby("Lens ID"):
        for i, j in itertools.permutations(d.Institution.values, 2): co.loc[i, j] += 1
    Wco = rs(co.values); rows = []
    for nm, W in (("floor25 (z, exclusive works)", Wm(DD[np.ix_(okx, okx)].copy(), "floor25")), ("excl_samecity (z, exclusive works)", Wm(DD[np.ix_(okx, okx)].copy(), "excl_samecity")),
                  ("co-authorship W (z, exclusive works)", rs(co.values[np.ix_(okx, okx)]))):
        I, p, _ = moran(zz[okx], W); rows.append(dict(spec=nm, n=int(okx.sum()), I=I, p=p))
    F2 = pd.DataFrame(rows); keep("F_noise_adjusted_moran", F2.round(3)); F2.to_csv(OUT/"F_noise_adjusted_moran.csv", index=False)
    print(f"※ 공저 연결이 있는 기관 쌍 수: {int((co.values[np.triu_indices(33, 1)] > 0).sum())}, 단독 논문 기준 기관 {int(okx.sum())}개. "
          "공저가 베이징 내부에 몰려 있으면 'core–periphery는 관계 구조에는 있으나 결과 유사도에는 없다'로, 없으면 부록 한 줄로 처리")

    # ---------- Fig 5 ----------
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8)); yi, lo_, hi_ = [], [], []
    for y in YEARS:
        s = cell[cell.year == y].set_index("Institution").y.reindex(INST); okk = ~s.isna().values
        if okk.sum() < 12: continue
        I, p, Ip = moran(s.values[okk], Wm(DD[np.ix_(okk, okk)].copy(), "floor25"), 999); yi.append((y, I)); lo_.append(np.percentile(Ip, 2.5)); hi_.append(np.percentile(Ip, 97.5))
    ax[0].fill_between([t[0] for t in yi], lo_, hi_, color=OI["grey"], alpha=.2, label="95% permutation band"); ax[0].plot(*zip(*yi), "o-", c=OI["blue"], ms=3.5, label="Moran's I (25 km floor)")
    ax[0].axhline(0, c="k", lw=.6); ax[0].axvline(EVENT, c="k", ls=":", lw=1); ax[0].set_xlabel("Publication year"); ax[0].set_title("(a) Moran's I by year", loc="left", fontweight="bold"); ax[0].legend(fontsize=7, frameon=False)
    bands = [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 3000)]; cI, cl_, ch_ = [], [], []
    for lo, hi in bands:
        W = ((DD > lo) & (DD <= hi)).astype(float); np.fill_diagonal(W, 0)
        if W.sum() == 0: cI.append(np.nan); cl_.append(np.nan); ch_.append(np.nan); continue
        I, p, Ip = moran(v0[ok0] if ok0.all() else np.where(ok0, v0, np.nanmean(v0)), rs(W), 999); cI.append(I); cl_.append(np.percentile(Ip, 2.5)); ch_.append(np.percentile(Ip, 97.5))
    xs = range(len(bands)); ax[1].fill_between(xs, cl_, ch_, color=OI["grey"], alpha=.2, label="95% permutation band"); ax[1].plot(xs, cI, "o-", c=OI["blue"], label="Moran's I (2013–19)")
    ax[1].set_xticks(list(xs)); ax[1].set_xticklabels([f"{l}–{h}" for l, h in bands]); ax[1].axhline(0, c="k", lw=.6); ax[1].set_xlabel("Distance band (km)"); ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_title("(b) Spatial correlogram, 2013–19", loc="left", fontweight="bold"); print(f"[Fig 5b] 0–50 km I = {cI[0]:.3f}, 순열 95% = [{cl_[0]:.3f}, {ch_[0]:.3f}] → 밴드 {'밖' if (cI[0] < cl_[0] or cI[0] > ch_[0]) else '안'}")
    G = np.abs(v0[:, None] - v0[None, :])[iu]; sm_ = same[iu]
    ax[2].scatter(DD[iu][~sm_], G[~sm_], s=12, c=OI["blue"], alpha=.5, label="Different city"); ax[2].scatter(DD[iu][sm_], G[sm_], s=14, c=OI["verm"], edgecolor="k", linewidth=.3, label="Same city")
    ax[2].set_xscale("log"); ax[2].set_xlabel("Distance (km, log scale)"); ax[2].set_ylabel("|Δ institution mean|"); ax[2].legend(fontsize=7, frameon=False)
    ax[2].set_title(f"(c) Outcome gap vs distance (ρ = {rho:.2f}, p = {p_rho:.2f}; n = {len(v0)})", loc="left", fontweight="bold", fontsize=9)
    plt.tight_layout(); plt.savefig(OUT/"fig_v9_F5_spatial_scale.png", dpi=300, bbox_inches="tight"); plt.close()

    # ---------- Fig 6 ----------
    if RUN_MAPS and GEOJSON.exists():
        import geopandas as gpd
        AEA = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs"
        prov = gpd.read_file(GEOJSON); prov = (prov.set_crs(4326) if prov.crs is None else prov).to_crs(AEA)
        prov["geometry"] = prov.geometry.simplify(2000); prov = prov.dissolve(by="NAME_1", as_index=False)
        pt = gpd.GeoSeries(gpd.points_from_xy(geo.lon, geo.lat), crs=4326).to_crs(AEA); geo["px"], geo["py"] = pt.x.values, pt.y.values
        P_ = {"2007-12": (2007, 2012), "2013-19": (2013, 2019), "2020-23": (2020, 2023)}
        devs = {p: outcome("A_avg_year_deviation", *r) for p, r in P_.items()}; vm = max(np.nanmax(np.abs(d.values)) for d in devs.values()); nrm = TwoSlopeNorm(0, -vm, vm)
        fig = plt.figure(figsize=(17, 11)); gs = fig.add_gridspec(2, 6, height_ratios=[1.35, 1]); axm = []
        for j, (p, (a_, b_)) in enumerate(P_.items()):
            ax_ = fig.add_subplot(gs[0, 2 * j:2 * j + 2]); axm.append(ax_); prov.plot(ax=ax_, color="#f4f2ec", edgecolor="#bdb9ae", linewidth=.3)
            nn = rows_iw[rows_iw.year.between(a_, b_)].groupby("Institution").size().reindex(INST).fillna(0).values
            ax_.scatter(geo.px, geo.py, s=25 + 260 * np.sqrt(nn / nn.max()), c=devs[p].values, cmap="coolwarm", norm=nrm, edgecolor="k", linewidth=.4, zorder=4)
            b = prov.total_bounds; ax_.set_xlim(b[0], b[2]); ax_.set_ylim(b[1], b[3]); ax_.set_aspect("equal"); ax_.set_axis_off()
            ax_.set_title(f"({'abc'[j]}) {p}: σ = {D1[p if p != '2020-23' else '2020-23']:.2f}", loc="left", fontweight="bold", fontsize=10)
            ia = ax_.inset_axes([0.02, 0.58, 0.30, 0.38]); bj = (geo.city == "Beijing").values
            ia.scatter(geo.px[bj], geo.py[bj], s=25 + 260 * np.sqrt(nn[bj] / nn.max()), c=devs[p].values[bj], cmap="coolwarm", norm=nrm, edgecolor="k", linewidth=.4)
            ia.set_xticks([]); ia.set_yticks([]); ia.set_title("Beijing (8)", fontsize=7, pad=2)
        fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap="coolwarm"), ax=axm, shrink=.55, label="Period-average deviation from year mean (log scale)")
        Pv = cell.pivot(index="Institution", columns="year", values="y").reindex(INST); bj = (geo.city == "Beijing").values
        ad = fig.add_subplot(gs[1, 0:3]); ad.plot(YEARS, np.nanstd(Pv.values[bj], axis=0, ddof=1), "o-", c=OI["verm"], ms=3.5, label=f"Beijing group (n = {int(bj.sum())})")
        ad.plot(YEARS, np.nanstd(Pv.values[~bj], axis=0, ddof=1), "s-", c=OI["blue"], ms=3.5, label=f"Other institutions (n = {int((~bj).sum())})")
        ad.axvline(EVENT, c="k", ls=":", lw=1); ad.set_xlabel("Publication year"); ad.set_ylabel("SD of institution means"); ad.legend(fontsize=8, frameon=False)
        ad.set_title("(d) Dispersion: Beijing group vs others", loc="left", fontweight="bold")
        bi, oi_ = np.where(bj)[0], np.where(~bj)[0]; gp = np.nanmean(Pv.values[bi], axis=0) - np.nanmean(Pv.values[oi_], axis=0)
        bs = np.array([np.nanmean(Pv.values[RNG.choice(bi, len(bi))], axis=0) - np.nanmean(Pv.values[RNG.choice(oi_, len(oi_))], axis=0) for _ in range(B_BOOT)])
        ae = fig.add_subplot(gs[1, 3:6]); ae.fill_between(YEARS, np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0), color=OI["grey"], alpha=.25, label="95% bootstrap interval (institutions)")
        ae.plot(YEARS, gp, "o-", c="k", ms=3.5, label="Mean gap: Beijing minus others"); ae.axhline(0, c="k", lw=.6); ae.axvline(EVENT, c="k", ls=":", lw=1)
        ae.set_xlabel("Publication year"); ae.set_ylabel("Mean gap (log scale)"); ae.legend(fontsize=8, frameon=False); ae.set_title("(e) Core–non-core mean gap", loc="left", fontweight="bold")
        plt.savefig(OUT/"fig_v9_F6_maps_core_periphery.png", dpi=300, bbox_inches="tight"); plt.close()
        print("[지도] GADM 경계에는 대만이 없습니다. Springer 지도 정책 및 분쟁지역 표기를 제출 전 확인하세요.")
    else: print("[지도 생략] RUN_MAPS=False 또는 경계 파일 없음")

# =========================== G. 일관성 보고서 + 저장 ====================
CR = pd.DataFrame(CHK); keep("G_consistency_report", CR, n=100)
print(f"\n일치 {int(CR.match.sum())}/{len(CR)} — match=False 항목은 원고 수치를 재계산 값으로 바꾸거나 정의를 본문에 명시")
for k, v in RES.items(): (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)).to_csv(OUT/f"{k}.csv", encoding="utf-8-sig")
(OUT/"analysis_manifest_v9.json").write_text(json.dumps({"seed": SEED, "n_draw": N_DRAW, "n_perm": N_PERM, "primary": PRIMARY, "n_papers": int(N),
    "bri_kw": BRI_KW, "tr_broad": TR_BROAD, "md5_family_links": hashlib.md5(FM.read_bytes()).hexdigest()}, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n완료:", OUT)

# ---- cell 02 ----
#

# ---- cell 03 ----
# v9b | 공저 가중치 강건성 (v9 셀 실행 직후, 같은 커널에서)
co_arr = co.values.astype(float); idx = np.where(okx)[0]; bjm = (geo.city.values == "Beijing")
def run_co(Wfull, nm):
    W = Wfull[np.ix_(idx, idx)].copy(); np.fill_diagonal(W, 0)
    I, p, _ = moran(zz[idx], rs(W))
    return dict(spec=nm, n=len(idx), n_pairs=int((W[np.triu_indices(len(idx), 1)] > 0).sum()), I=I, p=p)
nb = co_arr.copy(); nb[np.ix_(bjm, bjm)] = 0          # 베이징-베이징 쌍 제외
sc = co_arr.copy(); sc[same] = 0                       # 같은 도시 쌍 제외
rows = [run_co(co_arr, "count weights"), run_co((co_arr > 0).astype(float), "binary"),
        run_co((co_arr >= 3).astype(float), "binary, >=3 shared works"), run_co(np.log1p(co_arr), "log(1+count)"),
        run_co(nb, "excluding Beijing-Beijing pairs"), run_co(sc, "excluding same-city pairs")]
F3 = pd.DataFrame(rows); print(F3.round(3).to_string(index=False)); F3.to_csv(OUT/"F_coauthor_robustness.csv", index=False)
print("※ 6개 사양의 탐색적 검토. 같은 도시·베이징 내부 쌍을 빼도 I>0이고 p<.05이면 '거리와 별개의 관계 구조'로 한 줄 보고, "
      "빼면 사라지면 same-city 공저에 의존하는 것으로 쓰고 부록에서도 생략 가능")

# ---- cell 04 ----
# v9c | 유형 평균 제거 후 공저 Moran (v9 + v9b 직후, 같은 커널에서)
TYPE_CSV = BASE / "inst_type_33.csv"          # 열: Institution,type
if TYPE_CSV.exists():
    tp = pd.read_csv(TYPE_CSV).set_index("Institution").type.reindex(INST)
    ok_t = okx & tp.notna().values
    zt = pd.Series(zz[ok_t], index=np.array(INST)[ok_t]); zres = zt - zt.groupby(tp[ok_t].values).transform("mean")
    idx_t = np.where(ok_t)[0]; rows = []
    for nm, M in (("count weights", co_arr), ("log(1+count)", np.log1p(co_arr)), ("binary", (co_arr > 0).astype(float)),
                  ("excluding same-city pairs", np.where(same, 0, co_arr))):
        W = M[np.ix_(idx_t, idx_t)].copy(); np.fill_diagonal(W, 0); I, p, _ = moran(zres.values, rs(W))
        rows.append(dict(spec=nm, n=len(idx_t), I_after_type_demeaning=I, p=p))
    F4 = pd.DataFrame(rows); print(F4.round(3).to_string(index=False)); F4.to_csv(OUT/"F_coauthor_type_adjusted.csv", index=False)
    print("※ 유형 평균 제거 후에도 I>0이고 p<.05이면 '유형 동질성만으로는 설명되지 않음'으로 한 줄 추가. 사라지면 유형 동질성으로 해석하고 부록에서 생략 가능")
else:
    print("inst_type_33.csv가 없습니다 (열: Institution,type)")

# ---- cell 05 ----
# =====================================================================
# TransDiff-CN R1 | v10: 논문 게재용 Fig 1–4 (7.1 in 폭, 600 dpi PNG + PDF)
# 입력 : outputs_R1_v8/v8_01_cohort_compression.csv, outputs_R1_v9/{C_same_exposure,E_sigma_vs_noise}.csv,
#        outputs_R1_v9/D_null_lr.npz, data_R1/papers_unique.parquet, outputs_R1_v7/paper_family_links.csv.gz,
#        Transport_CN_Scholarly_Works.csv
# 출력 : figures_final/fig1..fig4 (.png/.pdf), captions_final.txt
# =====================================================================
import warnings, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

BASE = Path("~/Research/TransDiff-CN_R1").expanduser()
V8, V9 = BASE / "outputs_R1_v8", BASE / "outputs_R1_v9"
FIG = BASE / "figures_final"; FIG.mkdir(exist_ok=True)
POP, POPGZ = BASE / "data_R1/papers_unique.parquet", BASE / "data_R1/papers_unique.csv.gz"
FM, OLD = BASE / "outputs_R1_v7/paper_family_links.csv.gz", BASE / "Transport_CN_Scholarly_Works.csv"
EVENT, YEARS, HZ = 2013, list(range(2007, 2024)), 2023; NY = len(YEARS)
SEED = 20261122; RNG = np.random.default_rng(SEED); N_RES, N_PERM = 300, 200
W2 = 7.1                                   # 2단 폭(약 180 mm)
C = {"blk": "#000000", "blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "grey": "#777777"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "lines.linewidth": 1.1, "lines.markersize": 3, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False})

def title(ax, letter, text): ax.set_title(f"({letter}) {text}", loc="left", fontweight="bold")
def ref(ax, label=False):
    ax.axvline(EVENT, c="k", ls=":", lw=.7)
    if label: ax.text(EVENT + .2, ax.get_ylim()[0] + .02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "2013 (reference only)", fontsize=5.5, va="bottom")
def xfmt(ax): ax.set_xticks(range(2008, 2024, 4)); ax.set_xlabel("Publication year (cohort)")
def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    w, h = fig.get_size_inches() * 25.4; print(f"{name}: {w:.0f} x {h:.0f} mm"); plt.close(fig)

# ============================ Fig 1 ==================================
coh = pd.read_csv(V8 / "v8_01_cohort_compression.csv", index_col=0)
def line(ax, o, col, color, ls, mk, label):
    s = coh[coh.outcome == o].set_index("year"); x, y = s.index.values, s[col].values; small = s.cited_n.values < 100
    ax.plot(x, y, color=color, ls=ls, label=label)
    ax.plot(x[~small], y[~small], mk, color=color, ls="none"); ax.plot(x[small], y[small], mk, mfc="white", color=color, ls="none")
sx = pd.read_csv(V9 / "C_same_exposure.csv", index_col=0)
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.3)); a, b, c, d = axs.ravel()
line(a, "cum_any", "cited_share", C["blk"], "-", "o", "Cumulative count (Nov 2026)")
line(a, "any3_pri", "cited_share", C["blue"], "--", "s", "Within 3 years"); line(a, "any5_pri", "cited_share", C["verm"], "-.", "^", "Within 5 years")
a.set_ylabel("Share of works cited by patents"); title(a, "a", "Cited share by cohort"); a.legend(frameon=False, loc="upper left"); xfmt(a); ref(a, True)
line(b, "cum_any", "share_eq1", C["blk"], "-", "o", "Cumulative"); line(b, "any5_pri", "share_eq1", C["verm"], "-.", "^", "Within 5 years")
line(b, "nonCN5_pri", "share_eq1", C["green"], "--", "v", "5 years, excl. China-only families")
b.set_ylabel("Share with exactly one citing patent"); title(b, "b", "Floor effect among cited works"); b.legend(frameon=False, loc="lower right"); xfmt(b); ref(b)
line(c, "cum_any", "sd_log", C["blk"], "-", "o", "Cumulative"); line(c, "any5_pri", "sd_log", C["verm"], "-.", "^", "Within 5 years")
line(c, "nonCN5_pri", "sd_log", C["green"], "--", "v", "5 years, excl. China-only families")
c.set_ylabel("SD of log(1 + citing patents)"); title(c, "c", "Paper-level dispersion among cited works"); c.legend(frameon=False, loc="upper right"); xfmt(c); ref(c)
c.text(.02, .03, "Open markers: fewer than 100 cited works", transform=c.transAxes, fontsize=5.5)
for k, col_, ls_ in (("first_cite_within_1y", C["blue"], "-"), ("first_cite_within_2y", C["verm"], "--")):
    d.plot(sx.index, sx[k], ls_, color=col_, marker="o", label=f"First citing patent within {k.split('_')[-1][0]} year{'s' if k.endswith('2y') else ''}")
d.axvspan(2015.5, 2018.5, color=C["grey"], alpha=.14); d.axvspan(2020.5, 2023.5, color=C["grey"], alpha=.14)
d.text(2017, d.get_ylim()[1] * .97, "2016–18", ha="center", fontsize=5.5, va="top"); d.text(2022, d.get_ylim()[1] * .97, "2021–23", ha="center", fontsize=5.5, va="top")
d.set_ylabel("Share of works"); title(d, "d", "Equal-exposure comparison"); d.legend(frameon=False, loc="upper left"); xfmt(d); ref(d)
plt.tight_layout(); save(fig, "fig1_truncation_compression")

# ============================ 옛 33개 파일 =============================
old = pd.read_csv(OLD, encoding="utf-8-sig", encoding_errors="replace", low_memory=False); old.columns = old.columns.str.strip()
old = old[old["Lens ID"].notna() & (old["Lens ID"].astype(str) != "Lens ID")].copy()
old["year"] = pd.to_numeric(old["Publication Year"], errors="coerce"); old["cit"] = pd.to_numeric(old["Citing Patents Count"], errors="coerce")
old = old.dropna(subset=["year", "cit", "Institution"]); old["year"] = old.year.astype(int); old = old[old.year.between(2007, 2023)]
old["Institution"] = old.Institution.astype(str).str.replace(r"\s*\(.*\)$", "", regex=True).str.strip()
rows_iw = old.drop_duplicates(["Lens ID", "Institution"]).copy(); rows_iw["l"] = np.log1p(rows_iw.cit)

# ============================ Fig 2 ==================================
def icc1(cd, v):
    k, n, N_ = cd.max() + 1, np.bincount(cd), len(v)
    if k < 3 or N_ <= k: return np.nan
    gm, gmean = v.mean(), np.bincount(cd, weights=v) / n
    msb = (n * (gmean - gm) ** 2).sum() / (k - 1); msw = ((v - gmean[cd]) ** 2).sum() / (N_ - k); n0 = (N_ - (n ** 2).sum() / N_) / (k - 1)
    return max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))
rows = []
for y in YEARS:
    d_ = rows_iw[rows_iw.year == y]; cd, _ = pd.factorize(d_.Institution); v = d_.l.values
    nl = [icc1(cd, RNG.permutation(v)) for _ in range(N_PERM)]
    rows.append(dict(year=y, sd=v.std(ddof=1), icc=icc1(cd, v), lo=np.nanpercentile(nl, 2.5), hi=np.nanpercentile(nl, 97.5), ge3=(d_.cit.values >= 3).mean()))
F2 = pd.DataFrame(rows).set_index("year"); F2.to_csv(FIG / "fig2_data.csv")
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5))
axs[0].plot(F2.index, F2.sd, "o-", color=C["blk"]); axs[0].set_ylabel("SD of log(1 + citing patents)"); title(axs[0], "a", "Paper-level SD")
axs[1].fill_between(F2.index, F2.lo, F2.hi, color=C["blue"], alpha=.22, label="Label-permutation null (95%)")
axs[1].plot(F2.index, F2.icc, "o-", color=C["verm"], label="Observed ICC"); axs[1].set_ylabel("Between-institution share (ICC)"); title(axs[1], "b", "Variance between institutions"); axs[1].legend(frameon=False, loc="upper right")
axs[2].plot(F2.index, F2.ge3, "o-", color=C["blk"]); axs[2].set_ylabel("Share with ≥3 citing patents"); title(axs[2], "c", "Upper tail of paper counts")
for ax in axs: xfmt(ax); ref(ax)
ref(axs[0], True); plt.tight_layout(); save(fig, "fig2_variance_decomposition")

# ============================ Fig 3 ==================================
def sigma_matched(m):
    out = {}
    for y in YEARS:
        g = [v.cit.values for _, v in rows_iw[rows_iw.year == y].groupby("Institution") if len(v) >= m]
        if len(g) < 15: continue
        s = [np.log1p([RNG.choice(x, m, replace=False).mean() for x in g]).std(ddof=1) for _ in range(N_RES)]
        out[y] = (len(g), np.median(s), *np.percentile(s, [2.5, 97.5]))
    return pd.DataFrame(out, index=["k", "med", "lo", "hi"]).T
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7))
for m, col_ in ((3, C["blue"]), (5, C["verm"])):
    S = sigma_matched(m); axs[0].fill_between(S.index, S.lo, S.hi, color=col_, alpha=.18)
    axs[0].plot(S.index, S.med, "-", color=col_, label=f"{m} works per institution-year")
    full = S.k >= 33; axs[0].plot(S.index[full], S.med[full], "o", color=col_); axs[0].plot(S.index[~full], S.med[~full], "o", mfc="white", color=col_)
axs[0].set_ylabel("σ of institution means (log scale)"); title(axs[0], "a", "Fixed cell size (300 resamples)"); axs[0].legend(frameon=False)
axs[0].text(.02, .03, "Open markers: fewer than 33 institutions qualify", transform=axs[0].transAxes, fontsize=5.5)
E1 = pd.read_csv(V9 / "E_sigma_vs_noise.csv", index_col=0)
axs[1].fill_between(E1.index, E1.null_lo, E1.null_hi, color=C["blue"], alpha=.22, label="Label-permutation null (95%)")
axs[1].plot(E1.index, E1.sigma_obs, "-", color=C["verm"], label="Observed σ"); full = E1.k_inst >= 33
axs[1].plot(E1.index[full], E1.sigma_obs[full], "o", color=C["verm"]); axs[1].plot(E1.index[~full], E1.sigma_obs[~full], "o", mfc="white", color=C["verm"])
axs[1].set_ylabel("σ of institution means (log scale)"); title(axs[1], "b", "Observed σ vs sampling-noise benchmark"); axs[1].legend(frameon=False)
for ax in axs: xfmt(ax); ref(ax)
ref(axs[0], True); plt.tight_layout(); save(fig, "fig3_sigma_noise")

# ============================ Fig 4 ==================================
U = pd.read_parquet(POP) if POP.exists() else pd.read_csv(POPGZ, low_memory=False); U.columns = U.columns.str.strip()
U["year"] = pd.to_numeric(U["Publication Year"], errors="coerce"); U["cit"] = pd.to_numeric(U["Citing Patents Count"], errors="coerce")
U = U.dropna(subset=["year", "cit"]).reset_index(drop=True); U["year"] = U.year.astype(int); U = U[U.year.between(2007, 2023)].reset_index(drop=True)
ids, yidx = U["Lens ID"].astype(str), U.year.values - 2007
Fm = pd.read_csv(FM)
col_ = lambda nm: U[nm].fillna("").astype(str).str.lower() if nm in U.columns else pd.Series("", index=U.index)
txt = col_("Title") + " ; " + col_("Keywords") + " ; " + col_("Fields of Study")
BRI_RX = "|".join([r"railway", r"high-speed rail", r"rail transit", r"\bports?\b", r"shipping", r"maritime", r"container", r"freight", r"logistics",
                   r"belt and road", r"supply chain", r"intermodal", r"cross-border", r"corridor"])
mask_p = txt.str.contains(BRI_RX, regex=True).values
def wc(k):
    f = Fm[(Fm.pri <= HZ) & ((Fm.pri - Fm.py) <= k)]; return ids.map(f.groupby("paper_id").size()).fillna(0).values
OUTS = {"cumulative": (U.cit.values, YEARS[-1], "Cumulative count"), "3y": (wc(3), HZ - 3, "Cited within 3 years"), "5y": (wc(5), HZ - 5, "Cited within 5 years")}
NULL = np.load(V9 / "D_null_lr.npz"); fig, axs = plt.subplots(1, 3, figsize=(W2, 2.6)); chkv = {"cumulative": 0.586, "3y": 0.547, "5y": 0.521}
for j, (o, (cnt, last, lb)) in enumerate(OUTS.items()):
    c1 = (cnt >= 1).astype(float); v = np.array([y <= last for y in YEARS]); yrs = np.array(YEARS)[v]
    ng = np.bincount(yidx[mask_p], minlength=NY).astype(float); cg = np.bincount(yidx[mask_p], weights=c1[mask_p], minlength=NY)
    pn, pc = np.bincount(yidx, minlength=NY).astype(float), np.bincount(yidx, weights=c1, minlength=NY)
    lr = (np.log((cg + .5) / (ng + 1)) - np.log((pc - cg + .5) / (pn - ng + 1)))[v]; cgv = cg[v]
    i0 = int((yrs < EVENT).sum()); did = lr[yrs >= EVENT].mean() - lr[:i0].mean()
    print(f"[Fig 4 검증] {o}: DID = {did:.3f} (v9: {chkv[o]:.3f})")
    nl = NULL[f"all__{o}"]; q = np.percentile(nl, [5, 95], axis=0); b0 = lambda x: x - x[:i0].mean(); ax = axs[j]
    ax.fill_between(yrs, b0(q[0]), b0(q[1]), color=C["blue"], alpha=.22, label="Random topic groups (pointwise 5–95%)")
    ax.plot(yrs, b0(lr), "o-", color=C["verm"], label="BRI-related group")
    for y in (2012, 2013):
        k = list(yrs).index(y) if y in yrs else None
        if k is not None: ax.annotate(f"{y}: {int(cgv[k])} cited", (y, b0(lr)[k]), xytext=(4, -10 if y == 2013 else 6), textcoords="offset points", fontsize=5.5)
    ax.set_ylabel("Log ratio vs rest (pre-2013 mean = 0)" if j == 0 else ""); title(ax, "abc"[j], lb); xfmt(ax); ref(ax, j == 0)
    if j == 0: ax.legend(frameon=False, loc="upper left")
plt.tight_layout(); save(fig, "fig4_placebo_primary")

# ============================ 캡션 ==================================
cap = f"""Figure 1. Truncation and compression of patent-citation indicators, by publication cohort. (a) Share of works cited by at least one patent under the cumulative count (extracted November 2026) and under fixed 3- and 5-year windows (priority date of the citing family relative to publication year); windows are shown only for closed cohorts (to 2020 and 2018). (b) Share with exactly one citing patent and (c) SD of log(1 + citing patents), among cited works. Open markers: fewer than 100 cited works. (d) Share of works with a first citing patent within 1 and 2 years, shaded cohorts 2016–18 and 2021–23. Dotted line: 2013, for reference only.
Figure 2. Paper-level distribution and the between-institution share, 33-institution sample (institution–work rows; one row per work and institution). (a) SD of log(1 + citing patents). (b) Intraclass correlation by year with the 95% range under random reassignment of institution labels (ICC truncated at zero). (c) Share of works with three or more citing patents.
Figure 3. σ of institution means. (a) Cell size fixed at 3 and 5 works per institution-year (300 resamples; median and 95% range); open markers: fewer than 33 institutions qualify. (b) Observed σ against the 95% range from label permutations that preserve cell sizes (500 permutations); open markers: fewer than 33 institutions in the year.
Figure 4. Log ratio of the cited share of the primary BRI-related topic group relative to all other works (pre-2013 mean set to 0), by publication year, with the pointwise 5–95% band of 1,000 random topic groups of similar prevalence. Labels give the number of cited works in the group in 2012 and 2013. Inference rests on the difference-in-differences statistic in Table 2, not on the band. Dotted line: 2013, for reference only.
"""
(BASE / "captions_final.txt").write_text(cap, encoding="utf-8"); print("\n캡션 저장:", BASE / "captions_final.txt"); print("그림 저장:", FIG)

# ---- cell 06 ----
#

# ---- cell 07 ----
# =====================================================================
# TransDiff-CN R1 | v11: Fig 5(공간 규모), Fig 6(지도·코어-주변), Fig 7(공저 네트워크 지도)
# 입력 : Transport_CN_Scholarly_Works.csv, gadm41_CHN_1.json, (선택) inst_coords_33.csv
# 필요 : pandas numpy scipy matplotlib geopandas
# 출력 : figures_final/fig5_*, fig6_*, fig7_* (.png 600dpi + .pdf), captions_fig5to7.txt
# =====================================================================
import warnings, re, itertools
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import stats
import geopandas as gpd
warnings.filterwarnings("ignore")

BASE   = Path("~/Research/TransDiff-CN_R1").expanduser()
OLD    = BASE / "Transport_CN_Scholarly_Works.csv"
GEOJSON= BASE / "gadm41_CHN_1.json"
COORD  = BASE / "inst_coords_33.csv"            # 선택: 열 Institution,city,lat,lon (있으면 아래 K33을 덮어씀)
FIG    = BASE / "figures_final"; FIG.mkdir(exist_ok=True)
SEED = 20261122; RNG = np.random.default_rng(SEED)
N_PERM, N_PERM_YEAR, B_BOOT = 4999, 999, 1000   # 빠른 점검: 999, 199, 200
PERIOD, EVENT = (2013, 2019), 2013
YEARS = list(range(2007, 2024)); W2 = 7.1
C = {"blk": "#000000", "blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "grey": "#777777"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "lines.linewidth": 1.1, "lines.markersize": 3, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False})
def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    w, h = fig.get_size_inches() * 25.4; print(f"{name}: {w:.0f} x {h:.0f} mm"); plt.close(fig)
def ttl(ax, letter, text): ax.set_title(f"({letter}) {text}", loc="left", fontweight="bold")

# ----------------------------- 좌표 (근삿값: 제출 전 검증) ------------------------------
K33 = {"Chinese Academy of Sciences": ("Beijing", 39.98, 116.33), "Tsinghua University": ("Beijing", 40.00, 116.33), "Peking University": ("Beijing", 39.99, 116.31),
 "Beihang University": ("Beijing", 39.98, 116.35), "Beijing Institute of Technology": ("Beijing", 39.96, 116.31), "Beijing Jiaotong University": ("Beijing", 39.95, 116.34),
 "University of Chinese Academy of Sciences": ("Beijing", 39.91, 116.24), "Beijing University of Technology": ("Beijing", 39.87, 116.48), "Tianjin University": ("Tianjin", 39.11, 117.17),
 "Harbin Institute of Technology": ("Harbin", 45.75, 126.68), "Jilin University": ("Changchun", 43.83, 125.29), "Dalian University of Technology": ("Dalian", 38.89, 121.54),
 "Shandong University": ("Jinan", 36.67, 117.07), "Xi'an Jiaotong University": ("Xi'an", 34.25, 108.98), "Sichuan University": ("Chengdu", 30.63, 104.08),
 "Southwest Jiaotong University": ("Chengdu", 30.70, 104.00), "Chongqing University": ("Chongqing", 29.57, 106.47), "Central South University": ("Changsha", 28.17, 112.93),
 "University of Science and Technology of China": ("Hefei", 31.84, 117.26), "Nanjing University": ("Nanjing", 32.06, 118.78), "Southeast University": ("Nanjing", 32.06, 118.80),
 "Tongji University": ("Shanghai", 31.28, 121.50), "Fudan University": ("Shanghai", 31.30, 121.50), "Shanghai Jiao Tong University": ("Shanghai", 31.03, 121.44),
 "Shanghai University": ("Shanghai", 31.32, 121.40), "Zhejiang University": ("Hangzhou", 30.30, 120.09), "Wuhan University": ("Wuhan", 30.54, 114.36),
 "Wuhan University of Technology": ("Wuhan", 30.51, 114.33), "Huazhong University of Science and Technology": ("Wuhan", 30.51, 114.41),
 "South China University of Technology": ("Guangzhou", 23.15, 113.34), "Sun Yat-sen University": ("Guangzhou", 23.10, 113.30), "Shenzhen University": ("Shenzhen", 22.53, 113.93),
 "Soochow University": ("Suzhou", 31.30, 120.63)}
SHORT = {"Chinese Academy of Sciences": "CAS", "Tsinghua University": "THU", "Peking University": "PKU", "Beihang University": "BUAA",
         "Beijing Institute of Technology": "BIT", "Beijing Jiaotong University": "BJTU", "University of Chinese Academy of Sciences": "UCAS",
         "Beijing University of Technology": "BJUT"}
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()

# ----------------------------- 데이터 ------------------------------
old = pd.read_csv(OLD, encoding="utf-8-sig", encoding_errors="replace", low_memory=False); old.columns = old.columns.str.strip()
old = old[old["Lens ID"].notna() & (old["Lens ID"].astype(str) != "Lens ID")].copy()
old["year"] = pd.to_numeric(old["Publication Year"], errors="coerce"); old["cit"] = pd.to_numeric(old["Citing Patents Count"], errors="coerce")
old = old.dropna(subset=["year", "cit", "Institution"]); old["year"] = old.year.astype(int); old = old[old.year.between(2007, 2023)]
old["Institution"] = old.Institution.astype(str).str.replace(r"\s*\(.*\)$", "", regex=True).str.strip()
rows_iw = old.drop_duplicates(["Lens ID", "Institution"]).copy()
cell = rows_iw.groupby(["Institution", "year"]).cit.agg(mean="mean", n="size").reset_index()
cell["y"] = np.log1p(cell["mean"]); cell["dev"] = cell.y - cell.groupby("year").y.transform("mean")
INST = sorted(rows_iw.Institution.unique()); assert len(INST) == 33, f"기관 수 {len(INST)}"
K = {norm(k): v for k, v in K33.items()}
geo = pd.DataFrame({"Institution": INST}); geo[["city", "lat", "lon"]] = [K[norm(i)] for i in INST]; geo = geo.set_index("Institution")
if COORD.exists():
    ov = pd.read_csv(COORD).set_index("Institution"); geo.loc[ov.index.intersection(geo.index), ["city", "lat", "lon"]] = ov.loc[ov.index.intersection(geo.index), ["city", "lat", "lon"]]
    print("좌표 파일 적용:", COORD)
else: print("[주의] 좌표는 K33 근삿값입니다. 최종 좌표 파일(inst_coords_33.csv)이 있으면 덮어쓰세요.")
a, b = PERIOD
def hav(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    x = np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la)[:, None] * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(x))
DD = hav(geo.lat.values.astype(float), geo.lon.values.astype(float)); city = geo.city.values; same = city[:, None] == city[None, :]
rs = lambda W: W / W.sum(1, keepdims=True).clip(min=1e-12)
def Wm(D, kind, k=3):
    D = np.maximum(D.copy(), .5); n = len(D); np.fill_diagonal(D, np.inf)
    if kind == "floor25": W = 1 / np.maximum(D, 25)
    elif kind == "excl_samecity": W = np.where(D < 30, 0, 1 / np.maximum(D, 25))
    else:
        W = np.zeros((n, n))
        for i in range(n): W[i, np.argsort(D[i])[:k]] = 1
    np.fill_diagonal(W, 0); return rs(W)
def moran(z, W, nperm=N_PERM):
    z = np.asarray(z, float); z = z - z.mean(); n = len(z); S0 = W.sum(); den = z @ z; I = n / S0 * (z @ W @ z) / den
    Z = np.array([RNG.permutation(z) for _ in range(nperm)]); Ip = n / S0 * np.einsum("ri,ij,rj->r", Z, W, Z) / den
    return I, (1 + np.sum(np.abs(Ip - Ip.mean()) >= abs(I - Ip.mean()))) / (nperm + 1), Ip
avgdev = lambda a_, b_: cell[cell.year.between(a_, b_)].groupby("Institution").dev.mean().reindex(INST)
z0 = avgdev(a, b); v0 = z0.fillna(z0.mean()).values                 # 결과변수 A (v9와 동일)

# ===================================================================
# Fig 5: 공간 규모
# ===================================================================
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.6)); yi, nn, lo_, hi_ = [], [], [], []
for y in YEARS:
    s = cell[cell.year == y].set_index("Institution").y.reindex(INST); ok = ~s.isna().values
    if ok.sum() < 12: continue
    I, p, Ip = moran(s.values[ok], Wm(DD[np.ix_(ok, ok)], "floor25"), N_PERM_YEAR)
    yi.append((y, I)); nn.append(int(ok.sum())); lo_.append(np.percentile(Ip, 2.5)); hi_.append(np.percentile(Ip, 97.5))
xs_, I_ = zip(*yi); ax = axs[0]
ax.fill_between(xs_, lo_, hi_, color=C["grey"], alpha=.2, label="95% permutation band"); ax.plot(xs_, I_, "-", color=C["blue"], label="Moran's I (25 km floor)")
full = np.array(nn) >= 33; ax.plot(np.array(xs_)[full], np.array(I_)[full], "o", color=C["blue"]); ax.plot(np.array(xs_)[~full], np.array(I_)[~full], "o", mfc="white", color=C["blue"])
ax.axhline(0, c="k", lw=.5); ax.axvline(EVENT, c="k", ls=":", lw=.7); ax.set_xticks(range(2008, 2024, 4)); ax.set_xlabel("Publication year"); ax.set_ylabel("Moran's I")
ax.legend(frameon=False, loc="lower left"); ttl(ax, "a", "Moran's I by year"); ax.text(.02, .03 + .0, "", transform=ax.transAxes)
ax.text(.99, .02, "Open markers: fewer than 33 institutions", transform=ax.transAxes, ha="right", fontsize=5.5)
bands = [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 3000)]; cI, cl_, ch_, npairs = [], [], [], []
for lo, hi in bands:
    W = ((DD > lo) & (DD <= hi)).astype(float); np.fill_diagonal(W, 0); npairs.append(int(W[np.triu_indices(33, 1)].sum()))
    I, p, Ip = moran(v0, rs(W), N_PERM_YEAR); cI.append(I); cl_.append(np.percentile(Ip, 2.5)); ch_.append(np.percentile(Ip, 97.5))
ax = axs[1]; xs = np.arange(len(bands)); ax.fill_between(xs, cl_, ch_, color=C["grey"], alpha=.2, label="95% permutation band")
ax.plot(xs, cI, "o-", color=C["blue"], label="Moran's I, 2013–19"); ax.axhline(0, c="k", lw=.5)
ax.set_xticks(xs); ax.set_xticklabels([f"{l}–{h}\n(n={n})" for (l, h), n in zip(bands, npairs)]); ax.set_xlabel("Distance band, km (number of institution pairs)")
ttl(ax, "b", "Spatial correlogram, 2013–19"); ax.legend(frameon=False, loc="lower right")
print(f"[Fig 5b] 0–50 km: I = {cI[0]:.3f}, 95% = [{cl_[0]:.3f}, {ch_[0]:.3f}] → 밴드 {'밖' if (cI[0] < cl_[0] or cI[0] > ch_[0]) else '안'}")
iu = np.triu_indices(33, 1); gap = lambda v: np.abs(v[:, None] - v[None, :])[iu]
rho = stats.spearmanr(gap(v0), DD[iu])[0]; rp = np.array([stats.spearmanr(gap(RNG.permutation(v0)), DD[iu])[0] for _ in range(N_PERM)])
p_rho = (1 + np.sum(np.abs(rp) >= abs(rho))) / (len(rp) + 1); sm_ = same[iu]; G = gap(v0)
ax = axs[2]; ax.scatter(DD[iu][~sm_], G[~sm_], s=4, c=C["blue"], alpha=.5, label="Different city"); ax.scatter(DD[iu][sm_], G[sm_], s=6, c=C["verm"], edgecolor="k", linewidth=.25, label="Same city")
ax.set_xscale("log"); ax.set_xlabel("Distance (km, log scale)"); ax.set_ylabel("|Δ institution mean|"); ax.legend(frameon=False, loc="upper left")
ttl(ax, "c", f"Outcome gap vs distance (ρ = {rho:.2f}, p = {p_rho:.2f}; n = 33)")
plt.tight_layout(); save(fig, "fig5_spatial_scale")
mo25 = moran(v0, Wm(DD, "floor25"))[:2]; print(f"[검증] Fig 5c ρ = {rho:.3f}, p = {p_rho:.3f} (v9: −0.019, .789) | Moran floor25 I = {mo25[0]:.3f}, p = {mo25[1]:.3f} (v9: −0.142, .163)")

# ===================================================================
# 지도 도구
# ===================================================================
AEA = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs"
prov = gpd.read_file(GEOJSON); prov = (prov.set_crs(4326) if prov.crs is None else prov).to_crs(AEA)
prov["geometry"] = prov.geometry.simplify(2000); prov = prov.dissolve(by="NAME_1", as_index=False)
def project(lat, lon):
    g = gpd.GeoSeries(gpd.points_from_xy(np.asarray(lon, float), np.asarray(lat, float)), crs=4326).to_crs(AEA); return g.x.values, g.y.values
def base_map(ax):
    prov.plot(ax=ax, color="#f1efe8", edgecolor="#c9c5b8", linewidth=.25); bb = prov.total_bounds
    ax.set_xlim(bb[0], bb[2]); ax.set_ylim(bb[1], bb[3]); ax.set_aspect("equal"); ax.set_axis_off()
geo["px"], geo["py"] = project(geo.lat, geo.lon)
BJ = (geo.city == "Beijing").values; bj_names = [i for i in INST if geo.loc[i, "city"] == "Beijing"]
def circle_inset(ax, names, color_vals, sizes, norm_, cmap, edges=None, title_="Beijing (8), schematic layout", box=(0.0, 0.57, 0.36, 0.42)):
    ia = ax.inset_axes(list(box)); m = len(names); ang = 2 * np.pi * np.arange(m) / m + np.pi / 2; px, py = np.cos(ang), np.sin(ang)
    if edges is not None:
        mx = max(e[2] for e in edges) if edges else 1
        for i, j, w in edges: ia.plot([px[i], px[j]], [py[i], py[j]], "-", color=C["blue"], lw=.3 + 3.2 * np.sqrt(w / mx), alpha=.55, zorder=1)
    ia.scatter(px, py, s=sizes, c=color_vals, cmap=cmap, norm=norm_, edgecolor="k", linewidth=.4, zorder=3)
    for k, n_ in enumerate(names): ia.annotate(SHORT.get(n_, n_), (px[k], py[k]), xytext=(px[k] * 1.38, py[k] * 1.38), ha="center", va="center", fontsize=5)
    ia.set_xlim(-1.9, 1.9); ia.set_ylim(-1.9, 1.9); ia.set_xticks([]); ia.set_yticks([]); ia.set_title(title_, fontsize=5.5, pad=2)
    for s_ in ia.spines.values(): s_.set_linewidth(.5)
    return ia

# ===================================================================
# Fig 6: 기간별 지도와 코어-주변 비교
# ===================================================================
P_ = {"2007–12": (2007, 2012), "2013–19": (2013, 2019), "2020–23": (2020, 2023)}
sy = cell.groupby("year").y.std(ddof=1); devs = {p: avgdev(*r) for p, r in P_.items()}
vm = max(np.nanmax(np.abs(d.values)) for d in devs.values()); nrm = TwoSlopeNorm(0, -vm, vm)
fig = plt.figure(figsize=(W2, 5.7))
gs_t = fig.add_gridspec(1, 4, left=.01, right=.99, top=.95, bottom=.50, width_ratios=[1, 1, 1, .035], wspace=.04)
gs_b = fig.add_gridspec(1, 2, left=.075, right=.985, top=.38, bottom=.075, wspace=.30)
for j, (p, (a_, b_)) in enumerate(P_.items()):
    ax = fig.add_subplot(gs_t[0, j]); base_map(ax); nw = rows_iw[rows_iw.year.between(a_, b_)].groupby("Institution").size().reindex(INST).fillna(0).values
    sz = 8 + 85 * np.sqrt(nw / nw.max()); ax.scatter(geo.px, geo.py, s=sz, c=devs[p].values, cmap="coolwarm", norm=nrm, edgecolor="k", linewidth=.3, alpha=.9, zorder=4)
    s_ = sy.loc[a_:b_].mean(); ax.set_title(f"({'abc'[j]}) {p}: mean yearly σ = {s_:.2f}", loc="left", fontweight="bold", fontsize=7.5)
    bj_idx = np.where(BJ)[0]; circle_inset(ax, [INST[i] for i in bj_idx], devs[p].values[bj_idx], 8 + 85 * np.sqrt(nw[bj_idx] / nw.max()), nrm, "coolwarm")
cax = fig.add_subplot(gs_t[0, 3]); cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap="coolwarm"), cax=cax)
cb.set_label("Deviation from year mean\n(period average, log scale)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
Pv = cell.pivot(index="Institution", columns="year", values="y").reindex(INST); bi, oi_ = np.where(BJ)[0], np.where(~BJ)[0]
ad = fig.add_subplot(gs_b[0, 0]); ad.plot(YEARS, np.nanstd(Pv.values[bi], axis=0, ddof=1), "o-", color=C["verm"], label=f"Beijing group (n = {len(bi)})")
ad.plot(YEARS, np.nanstd(Pv.values[oi_], axis=0, ddof=1), "s--", color=C["blue"], label=f"Other institutions (n = {len(oi_)})")
ad.axvline(EVENT, c="k", ls=":", lw=.7); ad.set_xticks(range(2008, 2024, 4)); ad.set_xlabel("Publication year"); ad.set_ylabel("SD of institution means"); ad.legend(frameon=False); ttl(ad, "d", "Dispersion: Beijing group vs others")
gp = np.nanmean(Pv.values[bi], axis=0) - np.nanmean(Pv.values[oi_], axis=0)
bs = np.array([np.nanmean(Pv.values[RNG.choice(bi, len(bi))], axis=0) - np.nanmean(Pv.values[RNG.choice(oi_, len(oi_))], axis=0) for _ in range(B_BOOT)])
ae = fig.add_subplot(gs_b[0, 1]); ae.fill_between(YEARS, np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0), color=C["grey"], alpha=.25, label="95% bootstrap interval (institutions)")
ae.plot(YEARS, gp, "o-", color=C["blk"], label="Beijing minus others"); ae.axhline(0, c="k", lw=.5); ae.axvline(EVENT, c="k", ls=":", lw=.7)
ae.set_xticks(range(2008, 2024, 4)); ae.set_xlabel("Publication year"); ae.set_ylabel("Mean gap (log scale)"); ae.legend(frameon=False, loc="upper right"); ttl(ae, "e", "Core–non-core mean gap")
save(fig, "fig6_maps_core_periphery")
print("[지도] GADM 중국 경계에는 대만이 없습니다. Springer 지도 정책과 분쟁지역 표기를 제출 전 확인하세요.")

# ===================================================================
# Fig 7: 공저 네트워크 지도 (결과는 단독 기관 논문, 연결은 공저 논문)
# ===================================================================
wk = rows_iw[rows_iw.year.between(a, b)].copy(); wk["n_inst"] = wk.groupby("Lens ID").Institution.transform("nunique")
ex = wk[wk.n_inst == 1].assign(l=lambda d: np.log1p(d.cit)); s_w = ex.l.std(ddof=1)
g_ = ex.groupby("Institution").l.agg(["mean", "size"]).reindex(INST); okx = (g_["size"] >= 5).values
zz = ((g_["mean"] - ex.l.mean()) / (s_w / np.sqrt(g_["size"]))).values            # 표집잡음 표준화 잔차
co = pd.DataFrame(0.0, index=INST, columns=INST)
for _, d in wk[wk.n_inst > 1].groupby("Lens ID"):
    for i, j in itertools.permutations(d.Institution.values, 2): co.loc[i, j] += 1
co_arr = co.values; tri = np.triu_indices(33, 1); tot = co_arr[tri].sum(); intra = co_arr[tri][same[tri]].sum()
print(f"\n[공저 요약] 2013–19 기관 쌍 공저 연결 합 {int(tot)}, 같은 도시 쌍 {int(intra)} ({intra / tot:.1%}); 연결된 기관 쌍 {int((co_arr[tri] > 0).sum())}/{len(tri[0])}; 단독 논문 기준 기관 {int(okx.sum())}개")
# 도시 단위 집계
cdf = geo.assign(n_works=rows_iw[rows_iw.year.between(a, b)].groupby("Institution").size().reindex(INST).fillna(0).values, z=zz).groupby("city").agg(
    lat=("lat", "mean"), lon=("lon", "mean"), n_works=("n_works", "sum"), z=("z", "mean"), n_inst=("lat", "size"))
cdf["px"], cdf["py"] = project(cdf.lat, cdf.lon); cities = list(cdf.index); cc = {c_: i for i, c_ in enumerate(cities)}
E = {}
for i, j in zip(*tri):
    if city[i] != city[j] and co_arr[i, j] > 0: k = tuple(sorted((cc[city[i]], cc[city[j]]))); E[k] = E.get(k, 0) + co_arr[i, j]
inter = sum(E.values()); bj_share = sum(w for (i, j), w in E.items() if "Beijing" in (cities[i], cities[j])) / inter
print(f"[도시 간 공저] 도시 쌍 {len(E)}개, 합 {int(inter)}; 베이징이 포함된 비중 {bj_share:.1%} (허브형 여부 판단용)")
zn = TwoSlopeNorm(0, -np.nanmax(np.abs(zz)), np.nanmax(np.abs(zz)))
fig = plt.figure(figsize=(W2, 6.5))
gs1 = fig.add_gridspec(1, 3, left=.01, right=.99, top=.95, bottom=.45, width_ratios=[1.55, 1, .05], wspace=.06)
gs2 = fig.add_gridspec(1, 2, left=.075, right=.985, top=.37, bottom=.075, wspace=.55, width_ratios=[1, 1.15])
ax = fig.add_subplot(gs1[0, 0]); base_map(ax); mx = max(E.values())
for (i, j), w in sorted(E.items(), key=lambda t: t[1]):
    ax.plot([cdf.px[i], cdf.px[j]], [cdf.py[i], cdf.py[j]], "-", color=C["blue"], lw=.3 + 3.0 * np.sqrt(w / mx), alpha=.5, zorder=2, solid_capstyle="round")
ax.scatter(cdf.px, cdf.py, s=10 + 120 * np.sqrt(cdf.n_works / cdf.n_works.max()), c=cdf.z, cmap="coolwarm", norm=zn, edgecolor="k", linewidth=.4, zorder=4)
for c_, r in cdf.iterrows():
    if c_ in ("Beijing", "Shanghai", "Wuhan", "Guangzhou", "Chengdu", "Nanjing", "Harbin", "Xi'an"): ax.annotate(c_, (r.px, r.py), xytext=(5, -9), textcoords="offset points", fontsize=5)
refs = [w for w in (2, 10, 40) if w <= mx]
lg1 = [Line2D([0], [0], color=C["blue"], lw=.3 + 3.0 * np.sqrt(w / mx), alpha=.6, label=f"{w} co-authored works") for w in refs]
ax.legend(handles=lg1, frameon=False, loc="lower left", fontsize=5.5, title="Between-city ties", title_fontsize=5.5)
ttl(ax, "a", "City-level co-authorship, 2013–19")
ax = fig.add_subplot(gs1[0, 1]); ax.set_axis_off(); bn = bj_names; bidx = [INST.index(n) for n in bn]; m = len(bn); ang = 2 * np.pi * np.arange(m) / m + np.pi / 2
px, py = np.cos(ang), np.sin(ang); ed = [(i, j, co_arr[bidx[i], bidx[j]]) for i in range(m) for j in range(i + 1, m) if co_arr[bidx[i], bidx[j]] > 0]; mxb = max(e[2] for e in ed)
for i, j, w in sorted(ed, key=lambda t: t[2]): ax.plot([px[i], px[j]], [py[i], py[j]], "-", color=C["blue"], lw=.3 + 3.5 * np.sqrt(w / mxb), alpha=.55, zorder=1)
nwb = rows_iw[rows_iw.year.between(a, b)].groupby("Institution").size().reindex(bn).values
ax.scatter(px, py, s=15 + 150 * np.sqrt(nwb / nwb.max()), c=zz[bidx], cmap="coolwarm", norm=zn, edgecolor="k", linewidth=.4, zorder=3)
for k, n_ in enumerate(bn): ax.annotate(SHORT[n_], (px[k], py[k]), xytext=(px[k] * 1.3, py[k] * 1.3), ha="center", va="center", fontsize=6)
ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.set_aspect("equal"); ttl(ax, "b", "Beijing group (n = 8)")
ax.text(0, -1.62, f"Strongest tie: {int(mxb)} works", ha="center", fontsize=5.5)
cax = fig.add_subplot(gs1[0, 2]); cb = fig.colorbar(plt.cm.ScalarMappable(norm=zn, cmap="coolwarm"), cax=cax); cb.set_label("Noise-adjusted outcome z\n(single-institution works)", fontsize=5.5); cb.ax.tick_params(labelsize=5.5)
# (c) 쌍 단위 산점도
idx = np.where(okx)[0]; zs = zz[idx]; ii, jj = np.triu_indices(len(idx), 1); gp_ = np.abs(zs[ii] - zs[jj]); cnt = co_arr[np.ix_(idx, idx)][ii, jj]; sm = same[np.ix_(idx, idx)][ii, jj]
rho2 = stats.spearmanr(gp_, cnt)[0]; rp2 = []
for _ in range(N_PERM):
    zp = RNG.permutation(zs); rp2.append(stats.spearmanr(np.abs(zp[ii] - zp[jj]), cnt)[0])
p2 = (1 + np.sum(np.abs(rp2) >= abs(rho2))) / (len(rp2) + 1)
ax = fig.add_subplot(gs2[0, 0]); xj = np.log10(1 + cnt) + np.where(cnt == 0, RNG.normal(0, .02, len(cnt)), 0)
ax.scatter(xj[~sm], gp_[~sm], s=4, c=C["blue"], alpha=.45, label="Different city"); ax.scatter(xj[sm], gp_[sm], s=6, c=C["verm"], edgecolor="k", linewidth=.25, label="Same city")
bins = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2–4"), (5, 9, "5–9"), (10, 10 ** 6, "10+")]
for lo, hi, lb in bins:
    mm = (cnt >= lo) & (cnt <= hi)
    if mm.sum() >= 3: ax.plot(np.log10(1 + cnt[mm]).mean(), np.median(gp_[mm]), "D", color="k", ms=4, zorder=5)
ax.plot([], [], "D", color="k", ms=4, label="Bin median")
ax.set_xlabel("Co-authored works, log10(1 + count)"); ax.set_ylabel("|Δ z| (single-institution works)"); ax.legend(frameon=False, loc="upper right")
ttl(ax, "c", f"Pairs (ρ = {rho2:.2f}, p = {p2:.2f}; n = {len(idx)})")
# (d) Moran 점-구간
sub = lambda M: M[np.ix_(idx, idx)]; same_s = sub(same)
specs = [("Distance (25 km floor)", Wm(sub(DD), "floor25")), ("Distance, excl. same city", Wm(sub(DD), "excl_samecity")),
         ("Co-authorship (count)", rs(sub(co_arr))), ("Co-authorship (log)", rs(np.log1p(sub(co_arr)))),
         ("Co-authorship (binary)", rs((sub(co_arr) > 0).astype(float))), ("Co-authorship, excl. same city", rs(np.where(same_s, 0, sub(co_arr))))]
rows = []; ax = fig.add_subplot(gs2[0, 1])
for k, (nm, W) in enumerate(specs):
    I, p, Ip = moran(zs, W); rows.append(dict(spec=nm, n=len(idx), I=I, p=p)); col_ = C["verm"] if "Co-author" in nm else C["blue"]; y_ = len(specs) - 1 - k
    ax.hlines(y_, np.percentile(Ip, 2.5), np.percentile(Ip, 97.5), color=C["grey"], lw=3.5, alpha=.35); ax.plot(I, y_, "o", color=col_, ms=4)
    ax.text(.30, y_, f"p = {p:.3f}" if p >= .001 else "p < .001", va="center", fontsize=5.5)
ax.axvline(0, c="k", lw=.5); ax.set_yticks(range(len(specs))); ax.set_yticklabels([r["spec"] for r in rows][::-1], fontsize=5.5); ax.set_xlim(-.4, .45)
ax.set_xlabel("Moran's I (bar: 95% permutation range)"); ttl(ax, "d", "Outcome similarity by weight type")
save(fig, "fig7_coauthorship_network")
R7 = pd.DataFrame(rows); print("\n### Fig 7(d) 값"); print(R7.round(3).to_string(index=False)); R7.to_csv(FIG / "fig7_moran_values.csv", index=False)
print("※ v9b와 비교: 공저(count) I≈0.10(p≈.006), excl. same city I≈0.22. 크게 다르면 좌표나 기간 정의를 점검")

# ===================================================================
# 캡션
# ===================================================================
cap = """Figure 5. Spatial dependence in institution outcomes, 33 institutions. (a) Moran's I of annual institution means (inverse distance with a 25 km floor, row-standardised) with the 95% range from random permutation of institution labels; open markers: years with fewer than 33 institutions. (b) Correlogram for 2013–19 using binary distance-band weights; the number of institution pairs per band is given in parentheses; grey band: 95% permutation range. (c) Absolute outcome gap between institution pairs against distance, same-city pairs highlighted; Spearman correlation with permutation p-value. The outcome is the period-average deviation of log(1 + mean citing patents) from the year mean. Coordinates are approximate.
Figure 6. Dispersion of institution means across periods and the Beijing group. (a–c) Period-average deviation of each institution's outcome from the year mean; marker size is the number of works in the period; the Beijing insets use a schematic circular layout. Titles give the mean over years of the yearly σ across institutions. (d) Yearly SD of institution means for the Beijing group and all others. (e) Mean outcome gap (Beijing minus others) with a 95% bootstrap interval over institutions. Dotted line: 2013, for reference only. Map boundaries: GADM.
Figure 7. Co-authorship ties and outcome similarity, 2013–19 (exploratory). (a) City-level network: marker size is the number of institution-work rows, colour the mean noise-adjusted outcome z (single-institution works only); line width is the number of works co-authored between institutions in different cities. (b) Institution-level co-authorship within the Beijing group (schematic layout). (c) Absolute difference in z against the number of co-authored works for institution pairs; diamonds are bin medians; Spearman correlation with permutation p-value. (d) Moran's I of z under distance-based and co-authorship-based weights; bars give the 95% permutation range. Ties are measured among works cited by patents only and describe association, not knowledge flows.
"""
(BASE / "captions_fig5to7.txt").write_text(cap, encoding="utf-8"); print("\n캡션 저장:", BASE / "captions_fig5to7.txt"); print("그림 저장:", FIG)

# ---- cell 08 ----
#

# ---- cell 09 ----
# =====================================================================
# TransDiff-CN R1 | v12: 수치 확정 (원고 대조 원장 + 표 3종 + 치환 문장)
# 입력 : outputs_R1_v7/v8/v9 CSV, 모집단 파일, 옛 33개 기관 파일
# 출력 : outputs_R1_final/{H_ledger.csv, H_table1.md, H_table2.md, H_table3.md,
#        H_coauthor.md, H_text_fill.md, H_numbers.json, manifest}
# =====================================================================
import warnings, json, hashlib
from pathlib import Path
from contextlib import contextmanager
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

BASE = Path("~/Research/TransDiff-CN_R1").expanduser()
V7, V8, V9 = BASE/"outputs_R1_v7", BASE/"outputs_R1_v8", BASE/"outputs_R1_v9"
OUT = BASE/"outputs_R1_final"; OUT.mkdir(exist_ok=True)
POP, POPGZ = BASE/"data_R1/papers_unique.parquet", BASE/"data_R1/papers_unique.csv.gz"
FM, OLD = V7/"paper_family_links.csv.gz", BASE/"Transport_CN_Scholarly_Works.csv"
SEED = 20261122; RNG = np.random.default_rng(SEED)
N_FINAL, N_YEAR, N_RHO = 49999, 4999, 9999        # 빠른 점검: 4999, 999, 999
YEARS = list(range(2007, 2024))
TR_BROAD = (r"transport|traffic|railway|\brail\b|vehicle|\broads?\b|highway|logistic|shipping|aviation|airport|"
            r"\bports?\b|freight|transit|pavement|automotive|driving|maritime|mobility|intermodal")

# ------------------------------ 도구 ---------------------------------
@contextmanager
def block(name):
    try: yield
    except Exception as e: print(f"\n[{name}] 건너뜀: {type(e).__name__}: {e}")
LEDGER, NUM = [], {}
def L(id_, where, ms, new, tol, mode="eq", note=""):
    if new is None or (isinstance(new, float) and np.isnan(new)): ok = False
    elif mode == "le": ok = new <= ms + tol
    elif mode == "ge": ok = new >= ms - tol
    else: ok = abs(ms - new) <= tol
    LEDGER.append(dict(id=id_, where=where, manuscript=ms, recomputed=None if new is None else round(float(new), 4), mode=mode, tol=tol, match=bool(ok), note=note))
M = lambda x, d=3: f"{x:+.{d}f}".replace("-", "−")
Mu = lambda x, d=3: f"{x:.{d}f}".replace("-", "−")
P = lambda x: (f"{x:.3f}".lstrip("0") if x >= .001 else "<.001")
def md(df):
    cols = list(df.columns); s = "| " + " | ".join(map(str, cols)) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    for _, r in df.iterrows(): s += "| " + " | ".join(map(str, r.values)) + " |\n"
    return s
def rd(p, **k):
    d = pd.read_csv(p, **k); return d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
fmtI = lambda I, p: f"{Mu(I)} (p = {P(p)})"

# ------------------------------ 입력 로드 ----------------------------
A   = rd(V9/"A_sigma_definitions.csv", index_col=0); PSD = rd(V9/"A_paper_level_SD_33.csv", index_col=0)
B1  = rd(V9/"B_table1_contrast.csv"); C2 = rd(V9/"C_same_exposure_summary.csv"); C1 = rd(V9/"C_same_exposure.csv", index_col=0)
D1  = rd(V9/"D_sensitivity.csv"); SC = rd(V9/"D_event_year_scan.csv"); T2 = rd(V9/"D_table2_final.csv", index_col=[0, 1])
E1  = rd(V9/"E_sigma_vs_noise.csv", index_col=0); LOO = rd(V9/"E_2012_loo.csv", index_col=0)
T3c = rd(V9/"F_table3_definition_check.csv"); T3c = T3c[T3c.definition == "A_avg_year_deviation"]
COH = rd(V8/"v8_01_cohort_compression.csv", index_col=0)

# 옛 33개 기관 파일 (v9와 동일 정의)
old = pd.read_csv(OLD, encoding="utf-8-sig", encoding_errors="replace", low_memory=False); old.columns = old.columns.str.strip()
old = old[old["Lens ID"].notna() & (old["Lens ID"].astype(str) != "Lens ID")].copy()
old["year"] = pd.to_numeric(old["Publication Year"], errors="coerce"); old["cit"] = pd.to_numeric(old["Citing Patents Count"], errors="coerce")
old = old.dropna(subset=["year", "cit", "Institution"]); old["year"] = old.year.astype(int); old = old[old.year.between(2007, 2023)]
old["Institution"] = old.Institution.astype(str).str.replace(r"\s*\(.*\)$", "", regex=True).str.strip()
rows_iw = old.drop_duplicates(["Lens ID", "Institution"]).copy(); rows_iw["l"] = np.log1p(rows_iw.cit)
cell = rows_iw.groupby(["Institution", "year"]).cit.agg(mean="mean", n="size").reset_index()
cell["y"] = np.log1p(cell["mean"]); cell["dev"] = cell.y - cell.groupby("year").y.transform("mean"); INST = sorted(rows_iw.Institution.unique())
L("n institutions", "§3.1", 33, len(INST), 0); L("institution-work rows", "§3.1", 11271, len(rows_iw), 0)
L("unique works (33)", "§3.1", 9294, rows_iw["Lens ID"].nunique(), 0); L("non-empty cells", "§3.1", 541, len(cell), 0)

# ============ 1. σ와 논문 수준 SD ============
with block("σ·SD"):
    s = A["D1_mean_of_yearly_sigma"]
    NUM.update(sig_0712=s["2007-12"], sig_1319=s["2013-19"], sig_2023=s["2020-23"], sig_2123=s["2021-23"])
    NUM["dec_2023"], NUM["dec_2123"] = 1 - s["2020-23"] / s["2007-12"], 1 - s["2021-23"] / s["2007-12"]
    L("σ 2007–12", "§4.2", .478, s["2007-12"], .001); L("σ 2013–19", "§4.2", .266, s["2013-19"], .001)
    L("σ 2020–23", "§4.2", .126, s["2020-23"], .001); L("σ 2021–23", "§4.2", .105, s["2021-23"], .001)
    L("σ decline to 2021–23 (text '78.0%')", "§4.2/Highlights", .780, NUM["dec_2123"], .002)
    L("σ decline to 2020–23 (defined last period)", "§4.2", .780, NUM["dec_2023"], .002, note="정의된 마지막 기간 기준이면 73.6%: 본문 기간 표기와 퍼센트를 같은 기준으로 통일")
    pu, pr = PSD.unique_works, PSD.institution_work_rows
    NUM.update(psd_u07=pu[2007], psd_u23=pu[2023], psd_r07=pr[2007], psd_r23=pr[2023], dec_psd_r=1 - pr[2023] / pr[2007])
    L("paper SD unique 2007", "§4.2", .675, pu[2007], .001); L("paper SD unique 2023", "§4.2", .222, pu[2023], .001)
    L("paper SD rows 2007", "§4.2/Fig2a", .650, pr[2007], .001); L("paper SD rows 2023", "§4.2/Fig2a", .225, pr[2023], .001)
    L("paper SD decline rows ('about 65%')", "§4.2", .65, NUM["dec_psd_r"], .01)

def icc1(cd, v):
    k, n, N_ = cd.max() + 1, np.bincount(cd), len(v)
    if k < 3 or N_ <= k: return np.nan
    gm, gmean = v.mean(), np.bincount(cd, weights=v) / n
    msb = (n * (gmean - gm) ** 2).sum() / (k - 1); msw = ((v - gmean[cd]) ** 2).sum() / (N_ - k); n0 = (N_ - (n ** 2).sum() / N_) / (k - 1)
    return max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))
with block("ICC·상위꼬리"):
    icc, ge3 = {}, {}
    for y in YEARS:
        d = rows_iw[rows_iw.year == y]; cd, _ = pd.factorize(d.Institution); icc[y] = icc1(cd, d.l.values); ge3[y] = (d.cit.values >= 3).mean()
    NUM.update(icc_max=np.nanmax(list(icc.values())), icc_max_year=max(icc, key=lambda k: -1 if np.isnan(icc[k]) else icc[k]), ge3_07=ge3[2007], ge3_23=ge3[2023])
    L("ICC max (≤0.10)", "§4.2", .10, NUM["icc_max"], .005, "le"); L("share ≥3 citing, 2007", "§4.2/Fig2c", .38, ge3[2007], .01); L("share ≥3 citing, 2023", "§4.2/Fig2c", .03, ge3[2023], .01)

# ============ 2. 노출·절단 (Fig 1) ============
with block("코호트 지표"):
    g = lambda o, c, y: COH[(COH.outcome == o) & (COH.year == y)][c].iloc[0]
    cs = COH[COH.outcome == "cum_any"].set_index("year").cited_share; pk = cs.idxmax()
    NUM.update(cum_peak_year=int(pk), cum_peak=cs[pk], cum_2023=cs[2023], c5_2017=g("any5_pri", "cited_share", 2017), c5_2018=g("any5_pri", "cited_share", 2018))
    L("cumulative cited share, 2017 cohort ('11.5%')", "§4.1", .115, cs[2017], .003, note=f"피크 코호트 = {pk} ({cs[pk]:.3f})")
    L("cumulative cited share, 2023", "§4.1", .055, cs[2023], .003); L("5y share 2017", "§4.1", .10, NUM["c5_2017"], .006); L("5y share 2018", "§4.1", .10, NUM["c5_2018"], .006)
    s3 = COH[(COH.outcome == "any3_pri") & COH.year.between(2017, 2020)].cited_share
    L("3y share min 2017–20 (8–9%)", "§4.1", .08, s3.min(), .005, "ge"); L("3y share max 2017–20", "§4.1", .09, s3.max(), .005, "le")
    e5 = COH[(COH.outcome == "any5_pri") & (COH.year <= 2018)].share_eq1
    L("5y share exactly one: min", "§4.1", .63, e5.min(), .012); L("5y share exactly one: max", "§4.1", .72, e5.max(), .012)
    ce = COH[COH.outcome == "cum_any"].set_index("year").share_eq1; L("cumulative exactly-one 2007", "§4.1", .47, ce[2007], .01); L("cumulative exactly-one 2023", "§4.1", .83, ce[2023], .01)
    sd_c = COH[COH.outcome == "cum_any"].set_index("year").sd_log; L("pool paper SD 2007", "§4.2", .719, sd_c[2007], .002); L("pool paper SD 2023", "§4.2", .232, sd_c[2023], .002)
with block("같은 노출"):
    for _, r in C2.iterrows():
        L(f"same-exposure ratio lag {int(r.lag)}", "§4.1", {0: 1.14, 1: 1.01, 2: .94}[int(r.lag)], r.ratio, .006); NUM[f"ratio{int(r.lag)}"] = r.ratio; NUM[f"late{int(r.lag)}"] = r.late_cohorts

# ============ 3. Table 1 ============
MS_T1 = {("pool_all", "cumulative", "2007-12"): (-.135, -.189, -.080), ("pool_all", "cumulative", "2007-12 excl 2012"): (-.112, -.169, -.051), ("pool_tagged", "cumulative", "2007-12"): (-.044, -.118, .030),
 ("pool_all", "5y", "2007-12"): (-.107, -.173, -.040), ("pool_all", "5y", "2007-12 excl 2012"): (-.064, -.137, .003), ("pool_tagged", "5y", "2007-12"): (-.055, -.150, .066),
 ("pool_all", "3y", "2007-12"): (-.089, -.153, -.020), ("pool_all", "3y", "2007-12 excl 2012"): (-.041, -.104, .024), ("pool_tagged", "3y", "2007-12"): (-.051, -.151, .068),
 ("pool_all", "5y_nonCN", "2007-12"): (-.089, -.161, -.015), ("pool_all", "5y_nonCN", "2007-12 excl 2012"): (-.038, -.116, .038), ("pool_tagged", "5y_nonCN", "2007-12"): (-.041, -.158, .088),
 ("pool_nontagged", "cumulative", "2007-12"): (-.160, -.228, -.098)}
with block("Table 1"):
    for (sm, mt, en), (a_, lo, hi) in MS_T1.items():
        r = B1[(B1["sample"] == sm) & (B1.metric == mt) & (B1.early == en)].iloc[0]
        for nm, v, w in (("ΔSD", r.d_sd, a_), ("lo", r.sd_lo, lo), ("hi", r.sd_hi, hi)): L(f"T1 {sm}/{mt}/{en} {nm}", "Table 1", w, v, .0011)
    cell1 = lambda sm, mt, en="2007-12": (lambda r: f"{M(r.d_sd)} [{M(r.sd_lo)}, {M(r.sd_hi)}]")(B1[(B1["sample"] == sm) & (B1.metric == mt) & (B1.early == en)].iloc[0])
    LAB = {"cumulative": "Cumulative", "5y": "5-year window", "3y": "3-year window", "5y_nonCN": "5-year, excl. China-only families"}
    T1 = pd.DataFrame({"Metric": list(LAB.values()), "All, 2007–12 vs 2013–18": [cell1("pool_all", k) for k in LAB], "All, excl. 2012": [cell1("pool_all", k, "2007-12 excl 2012") for k in LAB],
                       "Transport-tagged": [cell1("pool_tagged", k) for k in LAB], "Non-tagged": [cell1("pool_nontagged", k) for k in LAB]})
    eq = lambda mt: B1[(B1["sample"] == "pool_all") & (B1.metric == mt) & (B1.early == "2007-12")].iloc[0]
    note = (f"Note. Share with exactly one citing patent rises by {Mu(eq('cumulative').d_eq1)} [{Mu(eq('cumulative').eq1_lo)}, {Mu(eq('cumulative').eq1_hi)}] (cumulative) and "
            f"{Mu(eq('5y').d_eq1)} [{Mu(eq('5y').eq1_lo)}, {Mu(eq('5y').eq1_hi)}] (5-year) in the full pool. Early/late cited works: {int(eq('cumulative').n_early):,}/{int(eq('cumulative').n_late):,} (cumulative), {int(eq('5y').n_early):,}/{int(eq('5y').n_late):,} (5-year).")
    L("T1 eq1 cumulative", "Table 1 note", .064, eq("cumulative").d_eq1, .0011); L("T1 eq1 5y", "Table 1 note", .041, eq("5y").d_eq1, .0011)
    (OUT/"H_table1.md").write_text(md(T1) + "\n" + note + "\n", encoding="utf-8"); NUM["T1"] = T1; print("\n### Table 1\n", md(T1))
    fx = lambda mt, en: B1[(B1["sample"] == "pool_all") & (B1.metric == mt) & (B1.early == en)].iloc[0]
    NUM["t1_5y_excl_ci"] = (fx("5y", "2007-12 excl 2012").sd_lo, fx("5y", "2007-12 excl 2012").sd_hi)

# ============ 4. Table 2, 민감도, 사건 연도 ============
MS_T2 = {("cumulative", "all"): (.586, -.578, .531, .027, .263, 47, 595), ("cumulative", "T_broad"): (.386, -.536, .465, .083, .234, 25, 317), ("5y", "all"): (.521, -.596, .662, .103, .900, 20, 134),
         ("5y", "T_broad"): (.333, -.661, .610, .249, .988, 11, 89), ("3y", "all"): (.547, -.663, .743, .139, .797, 15, 234), ("3y", "T_broad"): (.381, -.725, .745, .325, .914, 7, 136)}
with block("Table 2"):
    rows = []
    for (o, pop), (did, lo, hi, pe, pa, npre, npost) in MS_T2.items():
        r = T2.loc[(o, pop)]
        for nm, v, w, t in (("DID", r.observed_DID_long, did, .0011), ("q025", r.q025_DID_long, lo, .0011), ("q975", r.q975_DID_long, hi, .0011), ("p_emp", r.p_emp_DID_long, pe, .0011), ("p_adj", r.p_adj_DID_long, pa, .0011),
                            ("pre", r.cited_pre, npre, 0), ("post", r.cited_post, npost, 0)): L(f"T2 {o}/{pop} {nm}", "Table 2", w, v, t)
        rows.append({"Outcome": {"cumulative": "Cumulative", "5y": "5-year window", "3y": "3-year window"}[o], "Population": {"all": "All", "T_broad": "Transport-tagged"}[pop], "DID (log ratio)": Mu(r.observed_DID_long),
                     "Random-group 95% range": f"[{M(r.q025_DID_long)}, {M(r.q975_DID_long)}]", "p_emp": P(r.p_emp_DID_long), "p_adj": P(r.p_adj_DID_long), "BH p_emp / p_adj": f"{P(r.bh_p_emp)} / {P(r.bh_p_adj)}",
                     "Cited works pre / post 2013": f"{int(r.cited_pre)} / {int(r.cited_post)}", "DID excl. 2012": Mu(r.observed_DID_drop2012), "p_emp excl. 2012": P(r.p_emp_DID_drop2012)})
    T2o = pd.DataFrame(rows); (OUT/"H_table2.md").write_text(md(T2o) + "\nNote. The range and p_emp come from the same unadjusted distribution of 1,000 random topic groups; p_adj removes the dependence on pre-period cited share and pre-trend. BH = Benjamini–Hochberg across definitions.\n", encoding="utf-8")
    print("\n### Table 2\n", md(T2o)); NUM["T2o"] = T2o
    L("BH p_emp cumulative/all", "§4.3", .086, T2.loc[("cumulative", "all")].bh_p_emp, .0011); L("BH p_adj cumulative/all", "§4.3", .300, T2.loc[("cumulative", "all")].bh_p_adj, .0011)
    fw = T2.loc[[i for i in T2.index if i[0] in ("3y", "5y")]]
    NUM.update(fw_all_lo=T2.loc[[("3y", "all"), ("5y", "all")]].p_emp_DID_long.min(), fw_all_hi=T2.loc[[("3y", "all"), ("5y", "all")]].p_emp_DID_long.max(),
               fw_tag_lo=T2.loc[[("3y", "T_broad"), ("5y", "T_broad")]].p_emp_DID_long.min(), fw_tag_hi=T2.loc[[("3y", "T_broad"), ("5y", "T_broad")]].p_emp_DID_long.max())
    L("fixed-window p_emp min (text '.10')", "§4.3", .10, fw.p_emp_DID_long.min(), .005); L("fixed-window p_emp max (text '.25')", "§4.3", .25, fw.p_emp_DID_long.max(), .005, note="본문 '.10–.25'는 태그 표본 .325를 놓침 → '.10–.14(전체), .25–.33(태그)'로 수정")
    fxo = [i for i in T2.index if i[0] in ("3y", "5y", "5y_nonCN")]; L("cited_pre min fixed (primary)", "Highlights/§5.1", 7, T2.loc[fxo].cited_pre.min(), 0); L("cited_pre max fixed (primary)", "Highlights/§5.1", 20, T2.loc[fxo].cited_pre.max(), 0)
    cu = [i for i in T2.index if i[0] == "cumulative"]; L("cited_pre min cumulative", "§5.1", 25, T2.loc[cu].cited_pre.min(), 0); L("cited_pre max cumulative", "§5.1", 47, T2.loc[cu].cited_pre.max(), 0)
with block("민감도·MDE"):
    fxs = D1[D1.outcome.isin(["3y", "5y"])]; sig = fxs[fxs.p_emp < .05]; NUM["fx_sig"] = sig[["population", "outcome", "spec", "p_emp"]].values.tolist()
    L("fixed-window specs", "§4.3", 32, len(fxs), 0); L("fixed-window specs with p_emp<.05", "§4.3", 2, len(sig), 0, note="; ".join(f"{a}/{b}/{c} p={d:.3f}" for a, b, c, d in NUM["fx_sig"]))
    L("all spec: cumulative p<.05 (of 8)", "§5.1", 5, int(((D1.population == "all") & (D1.outcome == "cumulative") & (D1.p_emp < .05)).sum()), 0)
    mm = D1[D1.spec == "main(E=2013)"]; NUM.update(mde_main_lo=mm.MDE_rel_prob.min(), mde_main_hi=mm.MDE_rel_prob.max(), mde_fx_lo=mm[mm.outcome != "cumulative"].MDE_rel_prob.min(), mde_fx_hi=mm[mm.outcome != "cumulative"].MDE_rel_prob.max(), mde_all_hi=D1.MDE_rel_prob.max())
    L("MDE main rows min (text 1.5)", "§4.3", 1.5, NUM["mde_main_lo"], .06); L("MDE main rows max (text 2.5)", "§4.3", 2.5, NUM["mde_main_hi"], .35, note="주 사양 최대 2.2, 전체 사양 최대 %.2f" % NUM["mde_all_hi"])
    L("MDE fixed main min (1.9)", "§4.3", 1.9, NUM["mde_fx_lo"], .06); L("MDE fixed main max (2.2)", "§4.3", 2.2, NUM["mde_fx_hi"], .06)
    NUM["rank_lo"], NUM["rank_hi"] = SC.rank_of_2013_by_abs_z.min(), SC.rank_of_2013_by_abs_z.max(); L("2013 rank min (4)", "§4.3", 4, NUM["rank_lo"], 0); L("2013 rank max (6)", "§4.3", 6, NUM["rank_hi"], 0)
with block("v8 정의 전체의 사전 인용 논문 수"):
    V8p = rd(V8/"v8_04_placebo_full.csv"); V8p = V8p[V8p.stat == "DID_long"]
    fixd = V8p[V8p.outcome.isin(["any3_pri", "any5_pri", "nonCN5_pri"])].cited_pre; cumd = V8p[V8p.outcome == "cum_any"].cited_pre
    L("cited_pre across all definitions, fixed min (2)", "§5.1", 2, fixd.min(), 0); L("cited_pre across all definitions, fixed max (20)", "§5.1", 20, fixd.max(), 0)
    L("cited_pre across all definitions, cumulative min (13)", "§5.1", 13, cumd.min(), 0); L("cited_pre across all definitions, cumulative max (47)", "§5.1", 47, cumd.max(), 0)

# ============ 5. σ 노이즈 기준선 ============
with block("노이즈"):
    inside = ((E1.sigma_obs >= E1.null_lo) & (E1.sigma_obs <= E1.null_hi)); out_y = [int(y) for y in E1.index[~inside]]
    L("σ inside null 95% (of 17)", "§4.2", 16, int(inside.sum()), 0, note=f"밖: {out_y}")
    L("null σ 2007", "§4.2", .53, E1.null_mean[2007], .006); L("null σ 2023", "§4.2", .10, E1.null_mean[2023], .006)
    ab = E1.excess_var_share * E1.sigma_obs ** 2; NUM["excess_abs_2012"] = ab[2012]; NUM["excess_abs_max_other"] = ab.drop(2012).max()
    L("excess variance 2012 (~0.13)", "§4.2", .13, ab[2012], .01); L("excess variance, other years max (<0.05)", "§4.2", .05, ab.drop(2012).max(), 0, "le")
    for p_, (a, b) in {"2007-12": (2007, 2012), "2013-19": (2013, 2019), "2020-23": (2020, 2023)}.items():
        L(f"excess share {p_}", "§4.2", {"2007-12": .05, "2013-19": .13, "2020-23": .26}[p_], E1.loc[a:b, "excess_var_share"].mean(), .006)
    L("2012 σ", "§4.2", .688, E1.sigma_obs[2012], .001); L("2012 σ without UCAS", "§4.2", .590, LOO.sigma_without.min(), .001, note=str(LOO.sigma_without.idxmin()))
    NUM.update(out_years=out_y)
with block("Table A3 재현"):
    ym = cell.groupby("year").y.mean().loc[2008:2023]
    def aic_tab(s):
        yr = s.index.values.astype(float); t = yr - 2008; y = s.values; out = {}
        def aic(X): b = np.linalg.lstsq(X, y, rcond=None)[0]; r = ((y - X @ b) ** 2).sum(); n = len(y); return n * np.log(r / n) + 2 * X.shape[1]
        one = np.ones_like(t); out["Linear"] = aic(np.c_[one, t]); out["Quadratic"] = aic(np.c_[one, t, t ** 2])
        for E in range(2010, 2019): out[f"Hinge at {E}"] = aic(np.c_[one, t, np.maximum(0, yr - E)])
        return pd.Series(out)
    ta = pd.DataFrame({"All years": aic_tab(ym), "Excluding 2012": aic_tab(ym.drop(2012))}).round(1); print("\n### Table A3 재계산(AIC)\n", ta)
    ms_a3 = {"All years": {"Linear": -64.5, "Quadratic": -73.4, "Hinge at 2012": -77.1, "Hinge at 2013": -74.6}, "Excluding 2012": {"Linear": -66.3, "Quadratic": -79.3, "Hinge at 2013": -77.1, "Hinge at 2014": -78.6}}
    for c, d in ms_a3.items():
        for k, v in d.items():
            if k != "Linear": L(f"A3 ΔAIC vs linear {c}/{k}", "Table A3", v - d["Linear"], ta.loc[k, c] - ta.loc["Linear", c], 1.0, note="수준이 다르면 시리즈 정의가 다른 것: 원 스크립트로 재생성")
    ta.to_csv(OUT / "H_tableA3_recomputed.csv")

# ============ 6. 모집단·태그·커버리지 ============
with block("모집단"):
    U = pd.read_parquet(POP) if POP.exists() else pd.read_csv(POPGZ, low_memory=False); U.columns = U.columns.str.strip()
    U["year"] = pd.to_numeric(U["Publication Year"], errors="coerce"); U["cit"] = pd.to_numeric(U["Citing Patents Count"], errors="coerce")
    U = U.dropna(subset=["year", "cit"]); U["year"] = U.year.astype(int); U = U[U.year.between(2007, 2023)].reset_index(drop=True); ids = U["Lens ID"].astype(str)
    fos = U["Fields of Study"].fillna("").astype(str).str.lower(); tag = fos.str.contains(TR_BROAD, regex=True).values
    L("population works", "§3.1", 213328, len(U), 0); L("cited works", "§3.1", 16170, int((U.cit >= 1).sum()), 0); L("cited share", "§3.1", .076, (U.cit >= 1).mean(), .001)
    ty = pd.Series(tag).groupby(U.year.values).mean(); L("tag share 2007", "§3.1", .426, ty[2007], .001); L("tag share 2023", "§3.1", .187, ty[2023], .001); L("tag share pooled", "§3.1", .236, tag.mean(), .001)
    tmap = pd.Series(tag, index=ids.values); tmap = tmap[~tmap.index.duplicated()]; rw = rows_iw["Lens ID"].astype(str).map(tmap); uw = rows_iw.drop_duplicates("Lens ID")["Lens ID"].astype(str).map(tmap)
    L("tag share, 33-inst rows", "§3.1", .222, rw.mean(), .002); L("tag share, 33-inst unique works", "§3.1", .227, uw.mean(), .002)
    in33 = ids.isin(set(rows_iw["Lens ID"].astype(str))).values; cy = (U.cit >= 1).values
    cov = pd.Series(in33 & cy).groupby(U.year.values).sum() / pd.Series(cy).groupby(U.year.values).sum(); L("33-inst coverage of cited works 2007", "§4.2/§5.1", .92, cov[2007], .01); L("33-inst coverage 2023", "§4.2/§5.1", .35, cov[2023], .01)
    Fm = pd.read_csv(FM); Fm["cn_only"] = Fm["cn_only"].astype(str).str.lower().eq("true"); cn = Fm.groupby("py").cn_only.mean()
    L("China-only share 2007 cohort", "§4.2/§5.1", .07, cn.get(2007, np.nan), .02); L("China-only share 2018 cohort", "§4.2/§5.1", .37, cn.get(2018, np.nan), .02)

# ============ 7. 공간 통계: 순열 49,999회로 고정 (v11 커널 필요) ============
SP = all(k in globals() for k in ("DD", "same", "geo", "zz", "okx", "co_arr"))
def hav(la, lo):
    la, lo = np.radians(la), np.radians(lo); x = np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la)[:, None] * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2; return 6371 * 2 * np.arcsin(np.sqrt(x))
rs2 = lambda W: W / W.sum(1, keepdims=True).clip(min=1e-12)
def Wm2(D, kind, k=3):
    D = np.maximum(D.copy(), .5); n = len(D); np.fill_diagonal(D, np.inf)
    if kind == "nofloor": W = 1 / D
    elif kind == "floor25": W = 1 / np.maximum(D, 25)
    elif kind == "floor100": W = 1 / np.maximum(D, 100)
    elif kind == "excl_samecity": W = np.where(D < 30, 0, 1 / np.maximum(D, 25))
    else:
        W = np.zeros((n, n))
        for i in range(n): W[i, np.argsort(D[i])[:k]] = 1
    np.fill_diagonal(W, 0); return rs2(W)
def moran_fast(z, W, nperm=N_FINAL, chunk=5000):
    z = np.asarray(z, float); z = z - z.mean(); n = len(z); S0 = W.sum(); den = z @ z; I = n / S0 * (z @ W @ z) / den; Ip = []
    for s in range(0, nperm, chunk):
        m = min(chunk, nperm - s); Zs = z[np.argsort(RNG.random((m, n)), axis=1)]; Ip.append(n / S0 * np.einsum("ri,ij,rj->r", Zs, W, Zs) / den)
    Ip = np.concatenate(Ip); return I, (1 + np.sum(np.abs(Ip - Ip.mean()) >= abs(I - Ip.mean()))) / (nperm + 1), Ip
MS_T3 = {"nofloor": (-.262, .051), "floor25": (-.142, .163), "floor100": (-.076, .327), "excl_samecity": (0.000, .490), "knn3": (-.195, .179), "knn5": (-.098, .434), "city": (-.093, .647)}
LAB3 = {"nofloor": "Inverse distance, no floor", "floor25": "Inverse distance, 25 km floor", "floor100": "Inverse distance, 100 km floor", "excl_samecity": "Inverse distance, excluding same-city links (<30 km)",
        "knn3": "k nearest neighbours, k = 3", "knn5": "k nearest neighbours, k = 5", "city": "City-level aggregation, 25 km floor"}
if SP:
    with block("공간 Table 3"):
        z0 = cell[cell.year.between(2013, 2019)].groupby("Institution").dev.mean().reindex(INST).values; res = {}
        for kd in ("nofloor", "floor25", "floor100", "excl_samecity"): res[kd] = moran_fast(z0, Wm2(DD, kd))[:2]
        res["knn3"] = moran_fast(z0, Wm2(DD, "knn", 3))[:2]; res["knn5"] = moran_fast(z0, Wm2(DD, "knn", 5))[:2]
        cdf = pd.DataFrame({"c": geo.city.values, "z": z0, "la": geo.lat.values.astype(float), "lo": geo.lon.values.astype(float)}).groupby("c").mean()
        res["city"] = moran_fast(cdf.z.values, Wm2(hav(cdf.la.values, cdf.lo.values), "floor25"))[:2]; NUM["n_cities"] = len(cdf)
        for k, (I, p) in res.items(): L(f"T3 {k} I", "Table 3", MS_T3[k][0], I, .01); L(f"T3 {k} p", "Table 3", MS_T3[k][1], p, .02)
        Wn = Wm2(DD, "nofloor"); NUM["samecity_share"] = Wn[same].sum() / Wn.sum(); L("same-city weight share (no floor)", "§4.4", .568, NUM["samecity_share"], .005)
        L("n cities", "Table 3", 18, len(cdf), 0)
        T3o = pd.DataFrame([{"Specification": LAB3[k], "I": Mu(I), "p": P(p)} for k, (I, p) in res.items()]); (OUT/"H_table3.md").write_text(md(T3o) + f"\nNote. n = 33 (18 cities); row-standardised weights; {N_FINAL:,} permutations.\n", encoding="utf-8")
        print("\n### Table 3 (최종)\n", md(T3o)); NUM["T3"] = res; dist_I = [res[k][0] for k in ("nofloor", "floor25", "floor100", "excl_samecity", "knn3", "knn5")]
        NUM["moran_lo"], NUM["moran_hi"] = min(dist_I), max(dist_I)
        near = [(LAB3[k], round(p, 3)) for k, (I, p) in res.items() if .045 <= p <= .055]; print("※ p가 5% 경계(.045–.055)인 사양:", near or "없음")
        L("min p across Table 3 (not significant at 5%)", "§4.4/Abstract", .05, min(p for I, p in res.values()), 0, "ge", note="최소 p가 .05 미만이면 '어떤 사양도 5%에서 유의하지 않다' 문장 삭제")
    with block("공간 연도·구간·쌍"):
        out_n = 0; vals = []
        for y in YEARS:
            s = cell[cell.year == y].set_index("Institution").y.reindex(INST); ok = ~s.isna().values
            if ok.sum() < 12: continue
            I, p, Ip = moran_fast(s.values[ok], Wm2(DD[np.ix_(ok, ok)], "floor25"), N_YEAR); out_n += int(I < np.percentile(Ip, 2.5) or I > np.percentile(Ip, 97.5)); vals.append(y)
        L("years outside 95% band (Fig 5a)", "§4.4", 0, out_n, 0, note=f"검정 연도 {len(vals)}개")
        W = ((DD > 0) & (DD <= 50)).astype(float); np.fill_diagonal(W, 0); npairs = int(W[np.triu_indices(33, 1)].sum()); I, p, Ip = moran_fast(z0, rs2(W))
        lo_, hi_ = np.percentile(Ip, [2.5, 97.5]); NUM["b050"] = (I, lo_, hi_, npairs); L("0–50 km I", "§4.4", -.356, I, .02); L("0–50 km pairs", "§4.4", 40, npairs, 0)
        print(f"※ 0–50 km: I = {I:.3f}, 95% = [{lo_:.3f}, {hi_:.3f}] → 밴드 {'밖' if (I < lo_ or I > hi_) else '안'}")
        iu = np.triu_indices(33, 1); gap = lambda v: np.abs(v[:, None] - v[None, :])[iu]; rho = stats.spearmanr(gap(z0), DD[iu])[0]
        rp = np.array([stats.spearmanr(gap(RNG.permutation(z0)), DD[iu])[0] for _ in range(N_RHO)]); pr_ = (1 + np.sum(np.abs(rp) >= abs(rho))) / (N_RHO + 1); NUM["rho_gap"] = (rho, pr_)
        L("Spearman ρ gap-distance", "§4.4/Fig5c", -.02, rho, .01); L("Spearman p gap-distance", "§4.4/Fig5c", .78, pr_, .03)
    with block("공저 가중치"):
        idx = np.where(okx)[0]; sub = lambda Mx: Mx[np.ix_(idx, idx)]; bj = geo.city.values == "Beijing"; zs = zz[idx]; co_s = sub(co_arr)
        nb = co_arr.copy(); nb[np.ix_(bj, bj)] = 0; sc_ = np.where(same, 0, co_arr)
        specs = {"Distance (25 km floor)": Wm2(sub(DD), "floor25"), "Distance, excl. same city": Wm2(sub(DD), "excl_samecity"), "Co-authorship (count)": rs2(co_s), "Co-authorship (log)": rs2(np.log1p(co_s)),
                 "Co-authorship (binary)": rs2((co_s > 0).astype(float)), "Co-authorship, excl. same city": rs2(sub(sc_)), "Co-authorship, excl. Beijing–Beijing": rs2(sub(nb)), "Co-authorship (binary, ≥3 shared works)": rs2((co_s >= 3).astype(float))}
        rows = []
        for k, W in specs.items(): I, p, Ip = moran_fast(zs, W); rows.append(dict(spec=k, n=len(idx), I=I, p=p, lo=np.percentile(Ip, 2.5), hi=np.percentile(Ip, 97.5)))
        CO = pd.DataFrame(rows).set_index("spec"); NUM["CO"] = CO
        for k, (mi, mp) in {"Distance (25 km floor)": (-.184, .066), "Distance, excl. same city": (.024, .264), "Co-authorship (count)": (.101, .009), "Co-authorship (log)": (.068, .014),
                            "Co-authorship (binary)": (.044, .048), "Co-authorship, excl. same city": (.215, .000)}.items(): L(f"CO {k} I", "§4.5", mi, CO.loc[k, "I"], .01); L(f"CO {k} p", "§4.5", mp, CO.loc[k, "p"], .02)
        six = ["Co-authorship (count)", "Co-authorship (binary)", "Co-authorship (binary, ≥3 shared works)", "Co-authorship (log)", "Co-authorship, excl. Beijing–Beijing", "Co-authorship, excl. same city"]
        NUM["six_pos"], NUM["six_sig"] = int((CO.loc[six, "I"] > 0).sum()), int((CO.loc[six, "p"] < .05).sum()); L("six variants positive", "§4.5", 6, NUM["six_pos"], 0); L("six variants p<.05 ('four')", "§4.5", 4, NUM["six_sig"], 0, note="이진 가중 p가 .05 경계: 순열 횟수에 따라 4↔5로 바뀜")
        L("Beijing-excluded I (0.152)", "§4.5", .152, CO.loc["Co-authorship, excl. Beijing–Beijing", "I"], .01)
        ii, jj = np.triu_indices(len(idx), 1); gp_ = np.abs(zs[ii] - zs[jj]); cnt = co_s[ii, jj]; rho2 = stats.spearmanr(gp_, cnt)[0]
        rp2 = np.array([stats.spearmanr(np.abs(zp[ii] - zp[jj]), cnt)[0] for zp in (RNG.permutation(zs) for _ in range(N_RHO))]); p2 = (1 + np.sum(np.abs(rp2) >= abs(rho2))) / (N_RHO + 1)
        NUM["rho_co"] = (rho2, p2); L("Spearman ρ gap-coauthored", "§4.5", .01, rho2, .01); L("Spearman p gap-coauthored", "§4.5", .83, p2, .05)
        tri = np.triu_indices(33, 1); tot = co_arr[tri].sum(); L("co-author links total", "§4.5", 1103, tot, 0); L("same-city share of ties", "§4.5", .371, co_arr[tri][same[tri]].sum() / tot, .002)
        L("pairs with ties (of 528)", "§4.5", 311, int((co_arr[tri] > 0).sum()), 0)
        COo = CO.reset_index().assign(I=lambda d: d.I.map(Mu), p=lambda d: d.p.map(P), **{"95% permutation range": lambda d: [f"[{M(a)}, {M(b)}]" for a, b in zip(CO.lo, CO.hi)]})[["spec", "n", "I", "p", "95% permutation range"]]
        (OUT/"H_coauthor.md").write_text(md(COo) + f"\nNote. {N_FINAL:,} permutations; outcome = noise-standardised mean of single-institution works; 32 institutions with at least five such works. Exploratory; several specifications examined without correction.\n", encoding="utf-8"); print("\n### 공저 (최종)\n", md(COo))
else:
    print("\n[공간 블록 생략] v11을 같은 커널에서 먼저 실행하세요(DD, geo, zz, okx, co_arr 필요). 아래 Table 3은 v9 CSV 값(순열 4,999회)입니다.")
    k3 = {"nofloor": "nofloor", "floor25": "floor25", "floor100": "floor100", "excl_samecity": "excl_samecity", "knn3": "knn3", "knn5": "knn5", "city": "city_floor25"}
    T3o = pd.DataFrame([{"Specification": LAB3[k], "I": Mu(T3c[T3c.weights == v].I.iloc[0]), "p": P(T3c[T3c.weights == v].p.iloc[0])} for k, v in k3.items()]); (OUT/"H_table3.md").write_text(md(T3o), encoding="utf-8")

# ============ 8. 치환 문장 ============
with block("치환 문장"):
    n = NUM; T = n["T1"]; sig_list = "; ".join(f"{a}, {b}, {c} (p = {d:.3f})" for a, b, c, d in n["fx_sig"])
    co = n.get("CO"); T3 = n.get("T3", {})
    txt = f"""# 치환 문장 (모든 숫자는 이 셀의 재계산값)

## §4.2 σ 문단
수정 후: σ of institution means (yearly σ averaged over the period) falls from {Mu(n['sig_0712'])} (2007–12) to {Mu(n['sig_1319'])} (2013–19) and {Mu(n['sig_2023'])} (2020–23), a decline of {n['dec_2023']:.1%} between the first and last period ({Mu(n['sig_2123'])} in 2021–23 alone, {n['dec_2123']:.1%}).
→ Highlights/초록의 "about 78 percent"는 "about {round(n['dec_2023'] * 100)} percent"로 통일.

## §4.2 논문 수준 SD
수정 후: …the paper-level SD (institution–work rows) falls by about {n['dec_psd_r']:.0%} ({Mu(n['psd_r07'])} to {Mu(n['psd_r23'])}) and the share of works with three or more citing patents falls from {Mu(n['ge3_07'], 2)} to {Mu(n['ge3_23'], 2)}; the ICC lies between 0 and {Mu(n['icc_max'], 2)} in every year. Among unique cited works the SD falls from {Mu(n['psd_u07'])} to {Mu(n['psd_u23'])} (Fig. 2a uses rows; §4.2 uses unique works — state this in the caption).

## §4.2 노이즈 기준선
수정 후: Observed σ lies inside its 95% interval in 16 of 17 years (outside: {n['out_years']}); mean null σ is {Mu(E1.null_mean[2007], 2)} in 2007 and {Mu(E1.null_mean[2023], 2)} in 2023. Absolute excess variance is below {Mu(n['excess_abs_max_other'], 3)} in every year except 2012 ({Mu(n['excess_abs_2012'], 2)}).

## §4.1 노출
수정 후: Under the cumulative metric the share of works cited by patents peaks at {Mu(n['cum_peak'] * 100, 1)}% (cohort {n['cum_peak_year']}) and falls to {Mu(n['cum_2023'] * 100, 1)}% for 2023. At equal exposure the share with a first citing patent is similar for the latest observable cohorts and 2016–18 (lag 0: 2021–23; lag 1: 2021–22; lag 2: 2021; ratios {Mu(n['ratio0'], 2)}, {Mu(n['ratio1'], 2)}, {Mu(n['ratio2'], 2)}).

## §4.2 Table 1 (고정 창, 2012 제외)
수정 후: Under the 5-year window the paper-level SD falls by {Mu(-B1[(B1['sample']=='pool_all')&(B1.metric=='5y')&(B1.early=='2007-12')].iloc[0].d_sd)} between 2007–12 and 2013–18, but by {Mu(-B1[(B1['sample']=='pool_all')&(B1.metric=='5y')&(B1.early=='2007-12 excl 2012')].iloc[0].d_sd)} [{M(n['t1_5y_excl_ci'][0])}, {M(n['t1_5y_excl_ci'][1])}] when 2012 is excluded: the compression under fixed windows is sensitive to the 2012 cohort.

## §4.3 p 값
수정 후: Under fixed windows p_emp is {P(n['fw_all_lo'])}–{P(n['fw_all_hi'])} for all works and {P(n['fw_tag_lo'])}–{P(n['fw_tag_hi'])} for transport-tagged works (Table 2). In an independent Monte Carlo draw (Supplementary Table S1) fixed-window specifications reach p_emp < .05 in {len(n['fx_sig'])} of 32 cases: {sig_list}. The 2013 cohort has 3 cited works in the primary group under the 5-year window.
수정 후(MDE): random groups generate relative citation probabilities of about {Mu(n['mde_main_lo'], 1)}–{Mu(n['mde_main_hi'], 1)} in the main specifications ({Mu(n['mde_fx_lo'], 1)}–{Mu(n['mde_fx_hi'], 1)} under fixed windows; up to {Mu(n['mde_all_hi'], 1)} across all sensitivity specifications). 2013 ranks {int(n['rank_lo'])}–{int(n['rank_hi'])} of 7 among placebo event years.
"""
    if T3:
        I0, lo_, hi_, npr = n["b050"]; rho, pr_ = n["rho_gap"]
        txt += f"""
## §4.4 공간 (순열 {N_FINAL:,}회)
수정 후: Moran's I of institution means ranges from {Mu(n['moran_lo'], 2)} to {Mu(n['moran_hi'], 2)} across the distance-based weights (Table 3). Same-city links carry {n['samecity_share']:.1%} of the weight in the no-floor matrix; with a 25 km floor I = {fmtI(*T3['floor25'])}, excluding same-city links I = {fmtI(*T3['excl_samecity'])}, at the city level I = {fmtI(*T3['city'])}. In the 0–50 km band ({npr} pairs) I = {Mu(I0)}, 95% permutation range [{M(lo_)}, {M(hi_)}]. Pairwise gaps vs distance: Spearman ρ = {Mu(rho, 2)}, permutation p = {P(pr_)}.
"""
    if co is not None:
        r_, p_ = n["rho_co"]; g_ = lambda k: fmtI(co.loc[k, "I"], co.loc[k, "p"])
        txt += f"""
## §4.5 공저
수정 후: Moran's I is positive for co-authorship weights: {g_('Co-authorship (count)')} with counts, {g_('Co-authorship (log)')} with log counts, {g_('Co-authorship (binary)')} with a binary tie, and {g_('Co-authorship, excl. same city')} when same-city pairs are excluded; with distance weights it is {g_('Distance (25 km floor)')} and, excluding same-city pairs, {g_('Distance, excl. same city')}. In six weighting variants I is positive in {n['six_pos']} and p < .05 in {n['six_sig']}; without Beijing-internal pairs I = {Mu(co.loc['Co-authorship, excl. Beijing–Beijing', 'I'])}. Pairwise gaps vs number of co-authored works: ρ = {Mu(r_, 2)}, p = {P(p_)}.
"""
    (OUT / "H_text_fill.md").write_text(txt, encoding="utf-8"); print("\n치환 문장 저장:", OUT / "H_text_fill.md")

# ============ 9. 원장 출력·저장 ============
LG = pd.DataFrame(LEDGER); LG.to_csv(OUT / "H_ledger.csv", index=False, encoding="utf-8-sig")
bad = LG[~LG.match]; print(f"\n=== 원장: 일치 {int(LG.match.sum())}/{len(LG)} ===")
print(bad[["id", "where", "manuscript", "recomputed", "tol", "mode", "note"]].to_string(index=False) if len(bad) else "전부 일치")
for k in ("T1", "T2o", "T3", "CO", "b050", "rho_gap", "rho_co", "fx_sig"): NUM.pop(k, None)
(OUT / "H_numbers.json").write_text(json.dumps({k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in NUM.items()}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
(OUT / "manifest_v12.json").write_text(json.dumps({"seed": SEED, "n_final": N_FINAL, "n_year": N_YEAR, "n_rho": N_RHO, "spatial_block": bool(SP),
    "md5": {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(V9.glob("*.csv"))}}, indent=2), encoding="utf-8")
print("\n저장:", OUT)

# ---- cell 10 ----
#

# ---- cell 11 ----
out_years = []
for y in YEARS:
    s = cell[cell.year == y].set_index("Institution").y.reindex(INST); ok = ~s.isna().values
    if ok.sum() < 12: continue
    I, p, Ip = moran_fast(s.values[ok], Wm2(DD[np.ix_(ok, ok)], "floor25"), N_YEAR)
    if I < np.percentile(Ip, 2.5) or I > np.percentile(Ip, 97.5): out_years.append((y, round(I, 3), int(ok.sum())))
print(out_years)

# ---- cell 12 ----
V8p = rd(V8/"v8_04_placebo_full.csv"); V8p = V8p[(V8p.stat == "DID_long") & V8p.target.astype(str).str.startswith("BRI")]
print(V8p.groupby("outcome").cited_pre.agg(["min", "max"]))

# ---- cell 13 ----
strong = r"transport|railway|\brail\b|highway|\broads?\b|logistic|shipping|aviation|airport|\bports?\b|freight|pavement|automotive|maritime|intermodal"
only_amb = tag & ~fos.str.contains(strong, regex=True).values
print(f"tagged {tag.sum():,}; only ambiguous terms: {only_amb.sum():,} ({only_amb.sum()/tag.sum():.1%})")
print(fos[only_amb].str.extractall(r"(transit\w*|mobility\w*|driving\w*|traffic\w*|vehicle\w*)")[0].value_counts().head(15))

# ---- cell 14 ----
# =====================================================================
# TransDiff-CN R1 | 수정 필요 그림 일괄 재생성 v2 (단일 셀): Fig 1, 3, 4, 5, 7  [겹침 수정 반영]
# 입력: outputs_R1_v8, outputs_R1_v9, data_R1, outputs_R1_v7, 원본 CSV, GADM, (선택) inst_coords_33.csv
# 출력: figures_final/fig1,3,4,5,7 (.png/.pdf), captions_revised.txt, fig7_moran_values.csv
# =====================================================================
import warnings, re, itertools
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import stats
import geopandas as gpd
warnings.filterwarnings("ignore")

BASE = Path("~/Research/TransDiff-CN_R1").expanduser()
V8, V9 = BASE / "outputs_R1_v8", BASE / "outputs_R1_v9"
FIG = BASE / "figures_final"; FIG.mkdir(exist_ok=True)
POP, POPGZ = BASE / "data_R1/papers_unique.parquet", BASE / "data_R1/papers_unique.csv.gz"
FM, OLD = BASE / "outputs_R1_v7/paper_family_links.csv.gz", BASE / "Transport_CN_Scholarly_Works.csv"
GEOJSON, COORD = BASE / "gadm41_CHN_1.json", BASE / "inst_coords_33.csv"
EVENT, YEARS, HZ = 2013, list(range(2007, 2024)), 2023; NY = len(YEARS)
SEED = 20261122; RNG = np.random.default_rng(SEED)
N_RES, N_PERM_FIG, N_PERM_FINAL, B_BOOT = 300, 4999, 49999, 1000   # 빠른 점검: 499, 4999
PERIOD = (2013, 2019); a, b = PERIOD
W2 = 7.1
C = {"blk": "#000000", "blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "grey": "#777777"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "lines.linewidth": 1.1, "lines.markersize": 3, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False})

def ttl(ax, letter, text): ax.set_title(f"({letter}) {text}", loc="left", fontweight="bold")
def ref(ax, label=False):
    ax.axvline(EVENT, c="k", ls=":", lw=.7)
    if label:
        y0, y1 = ax.get_ylim(); ax.text(EVENT + .2, y0 + .02 * (y1 - y0), "2013 (reference only)", fontsize=5.5, va="bottom")
def xfmt(ax): ax.set_xticks(range(2008, 2024, 4)); ax.set_xlabel("Publication year (cohort)")
def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    w, h = fig.get_size_inches() * 25.4; print(f"{name}: {w:.0f} x {h:.0f} mm"); plt.close(fig)

# ------------------------------ 공통 데이터 (33개 기관) ------------------------------
old = pd.read_csv(OLD, encoding="utf-8-sig", encoding_errors="replace", low_memory=False); old.columns = old.columns.str.strip()
old = old[old["Lens ID"].notna() & (old["Lens ID"].astype(str) != "Lens ID")].copy()
old["year"] = pd.to_numeric(old["Publication Year"], errors="coerce"); old["cit"] = pd.to_numeric(old["Citing Patents Count"], errors="coerce")
old = old.dropna(subset=["year", "cit", "Institution"]); old["year"] = old.year.astype(int); old = old[old.year.between(2007, 2023)]
old["Institution"] = old.Institution.astype(str).str.replace(r"\s*\(.*\)$", "", regex=True).str.strip()
rows_iw = old.drop_duplicates(["Lens ID", "Institution"]).copy()
cell = rows_iw.groupby(["Institution", "year"]).cit.agg(mean="mean", n="size").reset_index()
cell["y"] = np.log1p(cell["mean"]); cell["dev"] = cell.y - cell.groupby("year").y.transform("mean")
INST = sorted(rows_iw.Institution.unique()); assert len(INST) == 33, f"기관 수 {len(INST)}"

# ------------------------------ 좌표 (근삿값: 제출 전 검증) ------------------------------
K33 = {"Chinese Academy of Sciences": ("Beijing", 39.98, 116.33), "Tsinghua University": ("Beijing", 40.00, 116.33), "Peking University": ("Beijing", 39.99, 116.31),
 "Beihang University": ("Beijing", 39.98, 116.35), "Beijing Institute of Technology": ("Beijing", 39.96, 116.31), "Beijing Jiaotong University": ("Beijing", 39.95, 116.34),
 "University of Chinese Academy of Sciences": ("Beijing", 39.91, 116.24), "Beijing University of Technology": ("Beijing", 39.87, 116.48), "Tianjin University": ("Tianjin", 39.11, 117.17),
 "Harbin Institute of Technology": ("Harbin", 45.75, 126.68), "Jilin University": ("Changchun", 43.83, 125.29), "Dalian University of Technology": ("Dalian", 38.89, 121.54),
 "Shandong University": ("Jinan", 36.67, 117.07), "Xi'an Jiaotong University": ("Xi'an", 34.25, 108.98), "Sichuan University": ("Chengdu", 30.63, 104.08),
 "Southwest Jiaotong University": ("Chengdu", 30.70, 104.00), "Chongqing University": ("Chongqing", 29.57, 106.47), "Central South University": ("Changsha", 28.17, 112.93),
 "University of Science and Technology of China": ("Hefei", 31.84, 117.26), "Nanjing University": ("Nanjing", 32.06, 118.78), "Southeast University": ("Nanjing", 32.06, 118.80),
 "Tongji University": ("Shanghai", 31.28, 121.50), "Fudan University": ("Shanghai", 31.30, 121.50), "Shanghai Jiao Tong University": ("Shanghai", 31.03, 121.44),
 "Shanghai University": ("Shanghai", 31.32, 121.40), "Zhejiang University": ("Hangzhou", 30.30, 120.09), "Wuhan University": ("Wuhan", 30.54, 114.36),
 "Wuhan University of Technology": ("Wuhan", 30.51, 114.33), "Huazhong University of Science and Technology": ("Wuhan", 30.51, 114.41),
 "South China University of Technology": ("Guangzhou", 23.15, 113.34), "Sun Yat-sen University": ("Guangzhou", 23.10, 113.30), "Shenzhen University": ("Shenzhen", 22.53, 113.93),
 "Soochow University": ("Suzhou", 31.30, 120.63)}
SHORT = {"Chinese Academy of Sciences": "CAS", "Tsinghua University": "THU", "Peking University": "PKU", "Beihang University": "BUAA",
         "Beijing Institute of Technology": "BIT", "Beijing Jiaotong University": "BJTU", "University of Chinese Academy of Sciences": "UCAS", "Beijing University of Technology": "BJUT"}
norm_ = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
K = {norm_(k): v for k, v in K33.items()}
geo = pd.DataFrame({"Institution": INST}); geo[["city", "lat", "lon"]] = [K[norm_(i)] for i in INST]; geo = geo.set_index("Institution")
if COORD.exists():
    ov = pd.read_csv(COORD).set_index("Institution"); ix = ov.index.intersection(geo.index); geo.loc[ix, ["city", "lat", "lon"]] = ov.loc[ix, ["city", "lat", "lon"]]; print("좌표 파일 적용:", COORD)
else: print("[주의] 좌표는 K33 근삿값입니다. 검증 후 inst_coords_33.csv로 덮어쓰세요.")
def hav(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    x = np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la)[:, None] * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(x))
DD = hav(geo.lat.values.astype(float), geo.lon.values.astype(float)); city = geo.city.values; same = city[:, None] == city[None, :]
rs = lambda W: W / W.sum(1, keepdims=True).clip(min=1e-12)
def Wm(D, kind, k=3):
    D = np.maximum(D.copy(), .5); n = len(D); np.fill_diagonal(D, np.inf)
    if kind == "floor25": W = 1 / np.maximum(D, 25)
    elif kind == "excl_samecity": W = np.where(D < 30, 0, 1 / np.maximum(D, 25))
    else:
        W = np.zeros((n, n))
        for i in range(n): W[i, np.argsort(D[i])[:k]] = 1
    np.fill_diagonal(W, 0); return rs(W)
def moran(z, W, nperm):
    z = np.asarray(z, float); z = z - z.mean(); n = len(z); S0 = W.sum(); den = z @ z; I = n / S0 * (z @ W @ z) / den
    Z = RNG.permuted(np.tile(z, (nperm, 1)), axis=1); Ip = n / S0 * np.einsum("ri,ij,rj->r", Z, W, Z) / den
    return I, (1 + np.sum(np.abs(Ip - Ip.mean()) >= abs(I - Ip.mean()))) / (nperm + 1), Ip
avgdev = lambda a_, b_: cell[cell.year.between(a_, b_)].groupby("Institution").dev.mean().reindex(INST)
z0 = avgdev(a, b); v0 = z0.fillna(z0.mean()).values

# ------------------------------ 전체 모집단 / 가족 링크 (Fig 1, 4) ------------------------------
U = pd.read_parquet(POP) if POP.exists() else pd.read_csv(POPGZ, low_memory=False); U.columns = U.columns.str.strip()
U["year"] = pd.to_numeric(U["Publication Year"], errors="coerce"); U["cit"] = pd.to_numeric(U["Citing Patents Count"], errors="coerce")
U = U.dropna(subset=["year", "cit"]).reset_index(drop=True); U["year"] = U.year.astype(int); U = U[U.year.between(2007, 2023)].reset_index(drop=True)
ids, yidx = U["Lens ID"].astype(str), U.year.values - 2007
Fm = pd.read_csv(FM)

# ============================ Fig 1 ==================================
coh = pd.read_csv(V8 / "v8_01_cohort_compression.csv", index_col=0)
def line(ax, o, col, color, ls, mk, label):
    s = coh[coh.outcome == o].set_index("year"); x, y = s.index.values, s[col].values; small = s.cited_n.values < 100
    ax.plot(x, y, color=color, ls=ls, label=label)
    ax.plot(x[~small], y[~small], mk, color=color, ls="none"); ax.plot(x[small], y[small], mk, mfc="white", color=color, ls="none")
first = Fm[Fm.pri <= HZ].assign(lag=lambda d: d.pri - d.py).groupby("paper_id").lag.min()
fl = ids.map(first); LAGS = (0, 1, 2); LATE = {0: (2021, 2023), 1: (2021, 2022), 2: (2021, 2021)}
SX = {k: (fl <= k).groupby(U.year).mean().loc[lambda s: s.index + k <= HZ] for k in LAGS}
for k in LAGS:
    r = SX[k].loc[LATE[k][0]:LATE[k][1]].mean() / SX[k].loc[2016:2018].mean()
    print(f"[Fig 1d] lag {k}: 최신/2016–18 비 = {r:.2f}  (본문: {dict(zip(LAGS, (1.14, 1.01, 0.94)))[k]})")
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.3)); A, B, Cc, D = axs.ravel()
line(A, "cum_any", "cited_share", C["blk"], "-", "o", "Cumulative (Nov 2026)")                      # [수정] 라벨 단축
line(A, "any3_pri", "cited_share", C["blue"], "--", "s", "Within 3 years"); line(A, "any5_pri", "cited_share", C["verm"], "-.", "^", "Within 5 years")
A.set_ylabel("Share of works cited by patents"); ttl(A, "a", "Cited share by cohort"); A.legend(frameon=False, loc="upper left", handlelength=1.6); xfmt(A); ref(A, True)
line(B, "cum_any", "share_eq1", C["blk"], "-", "o", "Cumulative"); line(B, "any5_pri", "share_eq1", C["verm"], "-.", "^", "Within 5 years")
line(B, "nonCN5_pri", "share_eq1", C["green"], "--", "v", "5 years, excl. China-only families")
B.set_ylabel("Share with exactly one citing patent"); ttl(B, "b", "Floor effect among cited works"); B.legend(frameon=False, loc="lower right"); xfmt(B); ref(B)
line(Cc, "cum_any", "sd_log", C["blk"], "-", "o", "Cumulative"); line(Cc, "any5_pri", "sd_log", C["verm"], "-.", "^", "Within 5 years")
line(Cc, "nonCN5_pri", "sd_log", C["green"], "--", "v", "5 years, excl. China-only families")
Cc.set_ylabel("SD of log(1 + citing patents)"); ttl(Cc, "c", "Paper-level dispersion among cited works"); Cc.legend(frameon=False, loc="upper right"); xfmt(Cc); ref(Cc)
Cc.text(.02, .03, "Open markers:\nfewer than 100 cited works", transform=Cc.transAxes, fontsize=5.5, va="bottom")   # [수정] 두 줄
LS = {0: (C["green"], "-", "^", "within 0 years"), 1: (C["blue"], "-", "o", "within 1 year"), 2: (C["verm"], "--", "s", "within 2 years")}
for k in LAGS:
    col_, ls_, mk_, lb_ = LS[k]; D.plot(SX[k].index, SX[k].values, ls_, color=col_, marker=mk_, label=lb_.capitalize())   # [수정] 범례 라벨 단축
D.axvspan(2015.5, 2018.5, color=C["grey"], alpha=.16, lw=0); D.axvspan(2020.5, 2023.5, color=C["grey"], alpha=.16, lw=0)   # [수정] 회색 하나
D.text(2017, .02, "2016–18", ha="center", va="bottom", fontsize=5.5, transform=D.get_xaxis_transform())                    # [수정] 아래로
D.text(2022, .02, "latest", ha="center", va="bottom", fontsize=5.5, transform=D.get_xaxis_transform())
D.set_ylabel("Share of works"); ttl(D, "d", "Equal-exposure comparison")
D.legend(frameon=False, loc="upper left", title="First citing patent", title_fontsize=6, handlelength=1.8); xfmt(D); ref(D)
plt.tight_layout(); save(fig, "fig1_truncation_compression")

# ============================ Fig 3 ==================================
def sigma_matched(m):
    out = {}
    for y in YEARS:
        g = [v.cit.values for _, v in rows_iw[rows_iw.year == y].groupby("Institution") if len(v) >= m]
        if len(g) < 15: continue
        s = [np.log1p([RNG.choice(x, m, replace=False).mean() for x in g]).std(ddof=1) for _ in range(N_RES)]
        out[y] = (len(g), np.median(s), *np.percentile(s, [2.5, 97.5]))
    return pd.DataFrame(out, index=["k", "med", "lo", "hi"]).T
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7))
for m, col_ in ((3, C["blue"]), (5, C["verm"])):
    S = sigma_matched(m); axs[0].fill_between(S.index, S.lo, S.hi, color=col_, alpha=.18); axs[0].plot(S.index, S.med, "-", color=col_, label=f"{m} works per institution-year")
    full = S.k >= 33; axs[0].plot(S.index[full], S.med[full], "o", color=col_); axs[0].plot(S.index[~full], S.med[~full], "o", mfc="white", color=col_)
axs[0].set_ylabel("σ of institution means (log scale)"); ttl(axs[0], "a", "Fixed cell size (300 resamples)"); axs[0].legend(frameon=False, loc="upper right")
axs[0].text(.02, .03, "Open markers:\nfewer than 33\ninstitutions qualify", transform=axs[0].transAxes, fontsize=5.5, va="bottom")   # [수정] 세 줄
E1 = pd.read_csv(V9 / "E_sigma_vs_noise.csv", index_col=0)
axs[1].fill_between(E1.index, E1.null_lo, E1.null_hi, color=C["blue"], alpha=.22, label="Label-permutation null (95%)")
axs[1].plot(E1.index, E1.sigma_obs, "-", color=C["verm"], label="Observed σ"); full = E1.k_inst >= 33
axs[1].plot(E1.index[full], E1.sigma_obs[full], "o", color=C["verm"]); axs[1].plot(E1.index[~full], E1.sigma_obs[~full], "o", mfc="white", color=C["verm"])
axs[1].set_ylabel("σ of institution means (log scale)"); ttl(axs[1], "b", "Observed σ vs sampling-noise benchmark"); axs[1].legend(frameon=False)
for ax in axs: xfmt(ax); ref(ax)
plt.tight_layout(); save(fig, "fig3_sigma_noise")

# ============================ Fig 4 ==================================
col_ = lambda nm: U[nm].fillna("").astype(str).str.lower() if nm in U.columns else pd.Series("", index=U.index)
txt = col_("Title") + " ; " + col_("Keywords") + " ; " + col_("Fields of Study")
BRI_RX = "|".join([r"railway", r"high-speed rail", r"rail transit", r"\bports?\b", r"shipping", r"maritime", r"container", r"freight", r"logistics",
                   r"belt and road", r"supply chain", r"intermodal", r"cross-border", r"corridor"])
mask_p = txt.str.contains(BRI_RX, regex=True).values
def wc(k):
    f = Fm[(Fm.pri <= HZ) & ((Fm.pri - Fm.py) <= k)]; return ids.map(f.groupby("paper_id").size()).fillna(0).values
OUTS = {"cumulative": (U.cit.values, YEARS[-1], "Cumulative count"), "3y": (wc(3), HZ - 3, "Cited within 3 years"), "5y": (wc(5), HZ - 5, "Cited within 5 years")}
OFF12 = {"cumulative": (8, 42), "3y": (-22, 40), "5y": (-30, 38)}             # (a): 오른쪽 위로, y축 눈금과 곡선을 피함
OFF13 = {"cumulative": (10, -16), "3y": (10, -16), "5y": (10, -16)}
NULL = np.load(V9 / "D_null_lr.npz"); fig, axs = plt.subplots(1, 3, figsize=(W2, 2.7)); chkv = {"cumulative": 0.586, "3y": 0.547, "5y": 0.521}
for j, (o, (cnt, last, lb)) in enumerate(OUTS.items()):
    c1 = (cnt >= 1).astype(float); v = np.array([y <= last for y in YEARS]); yrs = np.array(YEARS)[v]
    ng = np.bincount(yidx[mask_p], minlength=NY).astype(float); cg = np.bincount(yidx[mask_p], weights=c1[mask_p], minlength=NY)
    pn, pc = np.bincount(yidx, minlength=NY).astype(float), np.bincount(yidx, weights=c1, minlength=NY)
    lr = (np.log((cg + .5) / (ng + 1)) - np.log((pc - cg + .5) / (pn - ng + 1)))[v]; cgv = cg[v]
    i0 = int((yrs < EVENT).sum()); did = lr[yrs >= EVENT].mean() - lr[:i0].mean(); print(f"[Fig 4 검증] {o}: DID = {did:.3f} (v9: {chkv[o]:.3f})")
    nl = NULL[f"all__{o}"]; q = np.percentile(nl, [5, 95], axis=0); b0 = lambda x: x - x[:i0].mean(); ax = axs[j]
    ax.fill_between(yrs, b0(q[0]), b0(q[1]), color=C["blue"], alpha=.22, label="Random groups (5–95%)")           # [수정] 라벨 단축
    ax.plot(yrs, b0(lr), "o-", color=C["verm"], label="BRI-related group"); ax.margins(y=.22)
    for y, off in ((2012, OFF12[o]), (2013, OFF13[o])):
        k = list(yrs).index(y)
        ax.annotate(f"{y}: {int(cgv[k])} cited", (y, b0(lr)[k]), xytext=off, textcoords="offset points", fontsize=5.5, ha="center",
                    arrowprops=dict(arrowstyle="-", lw=.4, color="k"), zorder=6)
    ax.set_ylabel("Log ratio vs rest (pre-2013 mean = 0)" if j == 0 else ""); ttl(ax, "abc"[j], lb)
    ax.set_xticks(range(2008, last + 1, 4)); ax.set_xlim(2006.3, last + .7); ax.set_xlabel("Publication year (cohort)"); ref(ax)   # [수정] x축 끝 = 마지막 코호트
    if j == 0: ax.legend(frameon=False, loc="upper left", handlelength=1.5)
plt.tight_layout(); save(fig, "fig4_placebo_primary")

# ============================ Fig 5 ==================================
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.7)); yi, nn, lo_, hi_, outy = [], [], [], [], []
for y in YEARS:
    s = cell[cell.year == y].set_index("Institution").y.reindex(INST); ok = ~s.isna().values
    if ok.sum() < 12: continue
    I, p, Ip = moran(s.values[ok], Wm(DD[np.ix_(ok, ok)], "floor25"), N_PERM_FIG)
    l_, h_ = np.percentile(Ip, 2.5), np.percentile(Ip, 97.5); yi.append((y, I)); nn.append(int(ok.sum())); lo_.append(l_); hi_.append(h_)
    if I < l_ or I > h_: outy.append((y, round(float(I), 3), int(ok.sum())))
print(f"[Fig 5a] 4,999회 순열: 밴드 밖 연도 = {outy} → 본문 'inside in 16 of 17 years'와 대조 (기대: 2021 하나)")
xs_, I_ = zip(*yi); ax = axs[0]
ax.fill_between(xs_, lo_, hi_, color=C["grey"], alpha=.2); ax.plot(xs_, I_, "-", color=C["blue"])             # [수정] 범례 제거(캡션에 설명)
full = np.array(nn) >= 33; ax.plot(np.array(xs_)[full], np.array(I_)[full], "o", color=C["blue"]); ax.plot(np.array(xs_)[~full], np.array(I_)[~full], "o", mfc="white", color=C["blue"])
ax.axhline(0, c="k", lw=.5); ax.axvline(EVENT, c="k", ls=":", lw=.7); ax.set_xticks(range(2008, 2024, 4)); ax.set_xlabel("Publication year"); ax.set_ylabel("Moran's I")
ttl(ax, "a", "Moran's I by year")
bands = [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 3000)]; BL = ["0–50", "50–200", "200–500", "500–1k", "1–3k"]; cI, cl_, ch_, npairs = [], [], [], []
for lo, hi in bands:
    W = ((DD > lo) & (DD <= hi)).astype(float); np.fill_diagonal(W, 0); npairs.append(int(W[np.triu_indices(33, 1)].sum()))
    I, p, Ip = moran(v0, rs(W), N_PERM_FIG); cI.append(I); cl_.append(np.percentile(Ip, 2.5)); ch_.append(np.percentile(Ip, 97.5))
ax = axs[1]; xs = np.arange(len(bands)); ax.fill_between(xs, cl_, ch_, color=C["grey"], alpha=.2)
ax.plot(xs, cI, "o-", color=C["blue"]); ax.axhline(0, c="k", lw=.5)                                           # [수정] 범례 제거
ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(BL, npairs)], fontsize=5.2); ax.set_xlabel("Distance band, km (pairs)")   # [수정] 짧은 눈금
ttl(ax, "b", "Spatial correlogram, 2013–19")
print(f"[Fig 5b] 0–50 km: I = {cI[0]:.3f}, 95% = [{cl_[0]:.3f}, {ch_[0]:.3f}] (본문: −0.356, [−0.409, 0.395], 40 pairs → n = {npairs[0]})")
iu = np.triu_indices(33, 1); gap = lambda v: np.abs(v[:, None] - v[None, :])[iu]
rho = stats.spearmanr(gap(v0), DD[iu])[0]; rp = np.array([stats.spearmanr(gap(RNG.permutation(v0)), DD[iu])[0] for _ in range(N_PERM_FIG)])
p_rho = (1 + np.sum(np.abs(rp) >= abs(rho))) / (len(rp) + 1); sm_ = same[iu]; G = gap(v0)
ax = axs[2]; ax.scatter(DD[iu][~sm_], G[~sm_], s=4, c=C["blue"], alpha=.5, label="Different city"); ax.scatter(DD[iu][sm_], G[sm_], s=6, c=C["verm"], edgecolor="k", linewidth=.25, label="Same city")
ax.set_xscale("log"); ax.set_xlabel("Distance (km, log scale)"); ax.set_ylabel("|Δ institution mean|"); ax.set_ylim(top=G.max() * 1.3)
ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=.8, handletextpad=.2)
ax.text(.03, .80, f"ρ = {rho:.2f}, p = {p_rho:.2f}; n = 33", transform=ax.transAxes, ha="left", va="top", fontsize=5.8)    # 점이 없는 왼쪽 위로 이동
ttl(ax, "c", "Outcome gap vs distance")                                                                        # [수정] 짧은 제목
plt.tight_layout(w_pad=1.6); save(fig, "fig5_spatial_scale"); print(f"[Fig 5c] ρ = {rho:.3f}, p = {p_rho:.3f} (본문: ρ = −0.02, p = .78; 순열 오차 ±.01)")

# ============================ Fig 7 ==================================
AEA = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs"
prov = gpd.read_file(GEOJSON); prov = (prov.set_crs(4326) if prov.crs is None else prov).to_crs(AEA)
prov["geometry"] = prov.geometry.simplify(2000); prov = prov.dissolve(by="NAME_1", as_index=False)
def project(lat, lon):
    g = gpd.GeoSeries(gpd.points_from_xy(np.asarray(lon, float), np.asarray(lat, float)), crs=4326).to_crs(AEA); return g.x.values, g.y.values
def base_map(ax):
    prov.plot(ax=ax, color="#f1efe8", edgecolor="#c9c5b8", linewidth=.25); bb = prov.total_bounds
    ax.set_xlim(bb[0], bb[2]); ax.set_ylim(bb[1], bb[3]); ax.set_aspect("equal"); ax.set_axis_off()
geo["px"], geo["py"] = project(geo.lat, geo.lon); bj_names = [i for i in INST if geo.loc[i, "city"] == "Beijing"]
wk = rows_iw[rows_iw.year.between(a, b)].copy(); wk["n_inst"] = wk.groupby("Lens ID").Institution.transform("nunique")
ex = wk[wk.n_inst == 1].assign(l=lambda d: np.log1p(d.cit)); s_w = ex.l.std(ddof=1)
g_ = ex.groupby("Institution").l.agg(["mean", "size"]).reindex(INST); okx = (g_["size"] >= 5).values
zz = ((g_["mean"] - ex.l.mean()) / (s_w / np.sqrt(g_["size"]))).values
co = pd.DataFrame(0.0, index=INST, columns=INST)
for _, d in wk[wk.n_inst > 1].groupby("Lens ID"):
    for i, j in itertools.permutations(d.Institution.values, 2): co.loc[i, j] += 1
co_arr = co.values; tri = np.triu_indices(33, 1)
cdf = geo.assign(n_works=rows_iw[rows_iw.year.between(a, b)].groupby("Institution").size().reindex(INST).fillna(0).values, z=zz).groupby("city").agg(
    lat=("lat", "mean"), lon=("lon", "mean"), n_works=("n_works", "sum"), z=("z", "mean"), n_inst=("lat", "size"))
cdf["px"], cdf["py"] = project(cdf.lat, cdf.lon); cities = list(cdf.index); cc = {c_: i for i, c_ in enumerate(cities)}
E = {}
for i, j in zip(*tri):
    if city[i] != city[j] and co_arr[i, j] > 0: k = tuple(sorted((cc[city[i]], cc[city[j]]))); E[k] = E.get(k, 0) + co_arr[i, j]
zn = TwoSlopeNorm(0, -np.nanmax(np.abs(zz)), np.nanmax(np.abs(zz)))
fig = plt.figure(figsize=(W2, 6.5))
gs1 = fig.add_gridspec(1, 3, left=.01, right=.99, top=.95, bottom=.45, width_ratios=[1.55, 1, .05], wspace=.06)
gs2 = fig.add_gridspec(1, 2, left=.075, right=.985, top=.37, bottom=.075, wspace=.55, width_ratios=[1, 1.15])
ax = fig.add_subplot(gs1[0, 0]); base_map(ax); mx = max(E.values())
for (i, j), w in sorted(E.items(), key=lambda t: t[1]):
    ax.plot([cdf.px[i], cdf.px[j]], [cdf.py[i], cdf.py[j]], "-", color=C["blue"], lw=.3 + 3.0 * np.sqrt(w / mx), alpha=.5, zorder=2, solid_capstyle="round")
ax.scatter(cdf.px, cdf.py, s=10 + 120 * np.sqrt(cdf.n_works / cdf.n_works.max()), c=cdf.z, cmap="coolwarm", norm=zn, edgecolor="k", linewidth=.4, zorder=4)
LBL = {"Beijing": (-6, 8, "right"), "Shanghai": (8, -10, "left"), "Nanjing": (-6, 7, "right"), "Wuhan": (0, -13, "center"),      # 마커에 가깝게
       "Guangzhou": (8, -5, "left"), "Chengdu": (-5, -9, "right"), "Harbin": (7, -3, "left"), "Xi'an": (-6, -9, "right")}
for c_, r in cdf.iterrows():
    if c_ in LBL:
        dx, dy, ha_ = LBL[c_]; ax.annotate(c_, (r.px, r.py), xytext=(dx, dy), textcoords="offset points", fontsize=5, ha=ha_, zorder=6,
                                           bbox=dict(boxstyle="round,pad=.1", fc="#f1efe8", ec="none", alpha=.7))
lg1 = [Line2D([0], [0], color=C["blue"], lw=.3 + 3.0 * np.sqrt(w / mx), alpha=.6, label=f"{w} co-authored works") for w in (2, 10, 40) if w <= mx]
ax.legend(handles=lg1, frameon=False, loc="lower left", fontsize=5.5, title="Between-city ties", title_fontsize=5.5); ttl(ax, "a", "City-level co-authorship, 2013–19")
ax = fig.add_subplot(gs1[0, 1]); ax.set_axis_off(); bn = bj_names; bidx = [INST.index(n) for n in bn]; m = len(bn); ang = 2 * np.pi * np.arange(m) / m + np.pi / 2
px, py = np.cos(ang), np.sin(ang); ed = [(i, j, co_arr[bidx[i], bidx[j]]) for i in range(m) for j in range(i + 1, m) if co_arr[bidx[i], bidx[j]] > 0]; mxb = max(e[2] for e in ed)
for i, j, w in sorted(ed, key=lambda t: t[2]): ax.plot([px[i], px[j]], [py[i], py[j]], "-", color=C["blue"], lw=.3 + 3.5 * np.sqrt(w / mxb), alpha=.55, zorder=1)
nwb = rows_iw[rows_iw.year.between(a, b)].groupby("Institution").size().reindex(bn).values
ax.scatter(px, py, s=15 + 150 * np.sqrt(nwb / nwb.max()), c=zz[bidx], cmap="coolwarm", norm=zn, edgecolor="k", linewidth=.4, zorder=3)
for k, n_ in enumerate(bn): ax.annotate(SHORT[n_], (px[k], py[k]), xytext=(px[k] * 1.3, py[k] * 1.3), ha="center", va="center", fontsize=6)
ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.set_aspect("equal"); ttl(ax, "b", "Beijing group (n = 8)"); ax.text(0, -1.62, f"Strongest tie: {int(mxb)} works", ha="center", fontsize=5.5)
cax = fig.add_subplot(gs1[0, 2]); cb = fig.colorbar(plt.cm.ScalarMappable(norm=zn, cmap="coolwarm"), cax=cax); cb.set_label("Noise-adjusted outcome z\n(single-institution works)", fontsize=5.5); cb.ax.tick_params(labelsize=5.5)
idx = np.where(okx)[0]; zs = zz[idx]; ii, jj = np.triu_indices(len(idx), 1); gp_ = np.abs(zs[ii] - zs[jj]); cnt = co_arr[np.ix_(idx, idx)][ii, jj]; sm = same[np.ix_(idx, idx)][ii, jj]
rho2 = stats.spearmanr(gp_, cnt)[0]; rp2 = [stats.spearmanr(np.abs((zp := RNG.permutation(zs))[ii] - zp[jj]), cnt)[0] for _ in range(N_PERM_FIG)]
p2 = (1 + np.sum(np.abs(rp2) >= abs(rho2))) / (len(rp2) + 1)
ax = fig.add_subplot(gs2[0, 0]); xj = np.log10(1 + cnt) + np.where(cnt == 0, RNG.normal(0, .02, len(cnt)), 0)
ax.scatter(xj[~sm], gp_[~sm], s=4, c=C["blue"], alpha=.45, label="Different city"); ax.scatter(xj[sm], gp_[sm], s=6, c=C["verm"], edgecolor="k", linewidth=.25, label="Same city")
for lo, hi in [(0, 0), (1, 1), (2, 4), (5, 9), (10, 10 ** 6)]:
    mm = (cnt >= lo) & (cnt <= hi)
    if mm.sum() >= 3: ax.plot(np.log10(1 + cnt[mm]).mean(), np.median(gp_[mm]), "D", color="k", ms=4, zorder=5)
ax.plot([], [], "D", color="k", ms=4, label="Bin median"); ax.set_xlabel("Co-authored works, log10(1 + count)"); ax.set_ylabel("|Δ z| (single-institution works)")
ax.legend(frameon=False, loc="upper right"); ttl(ax, "c", f"Pairs (ρ = {rho2:.2f}, p = {p2:.2f}; n = {len(idx)})")
sub = lambda M: M[np.ix_(idx, idx)]; same_s = sub(same)
specs = [("Distance (25 km floor)", Wm(sub(DD), "floor25")), ("Distance, excl. same city", Wm(sub(DD), "excl_samecity")),
         ("Co-authorship (count)", rs(sub(co_arr))), ("Co-authorship (log)", rs(np.log1p(sub(co_arr)))),
         ("Co-authorship (binary)", rs((sub(co_arr) > 0).astype(float))), ("Co-authorship, excl. same city", rs(np.where(same_s, 0, sub(co_arr))))]
rows = []; ax = fig.add_subplot(gs2[0, 1])
for k, (nm, W) in enumerate(specs):
    I, p, Ip = moran(zs, W, N_PERM_FINAL); rows.append(dict(spec=nm, n=len(idx), I=I, p=p)); col_ = C["verm"] if "Co-author" in nm else C["blue"]; y_ = len(specs) - 1 - k
    ax.hlines(y_, np.percentile(Ip, 2.5), np.percentile(Ip, 97.5), color=C["grey"], lw=3.5, alpha=.35); ax.plot(I, y_, "o", color=col_, ms=4)
    ax.text(.30, y_, f"p = {p:.3f}" if p >= .001 else "p < .001", va="center", fontsize=5.5)
ax.axvline(0, c="k", lw=.5); ax.set_yticks(range(len(specs))); ax.set_yticklabels([r["spec"] for r in rows][::-1], fontsize=5.5); ax.set_xlim(-.4, .45)
ax.set_xlabel("Moran's I (bar: 95% permutation range)"); ttl(ax, "d", "Outcome similarity by weight type")
save(fig, "fig7_coauthorship_network")
R7 = pd.DataFrame(rows); print("\n### Fig 7(d) 값 (49,999회)"); print(R7.round(3).to_string(index=False)); R7.to_csv(FIG / "fig7_moran_values.csv", index=False)
print("→ 본문/Table A6 대조: count .101 (.008), log .068 (.013), binary .044 (.048), excl.same .215 (<.001), 거리 −.184 (.069), 거리excl .024 (.270). binary p가 .05 근처이므로 반드시 확인")

# ============================ 수정 캡션 ==================================
cap = """Figure 1 (a). (a) Share of works cited by at least one patent under the cumulative count (extracted November 2026) and under fixed 3- and 5-year windows (priority date of the citing family relative to publication year); windows are shown only for closed cohorts (to 2020 and 2018).
Figure 1 (d). (d) Share of works with a first citing patent within 0, 1 and 2 years, by cohort, for cohorts whose window is closed; shaded: cohorts 2016–18 and the latest observable cohorts (2021–23 at lag 0, 2021–22 at lag 1, 2021 at lag 2). Dotted line: 2013, for reference only.
Figure 3. [끝에 추가] Dotted line: 2013, for reference only.
Figure 4. [본문 유지; 범례 라벨이 "Random groups (5–95%)"로 줄었으므로 캡션에 "Random groups: 1,000 random topic groups of similar prevalence (pointwise 5–95% band)"가 이미 있는지 확인] Dotted line: 2013, for reference only.
Figure 5. Spatial dependence in institution outcomes, 33 institutions. (a) Moran's I of annual institution means (blue line; inverse distance with a 25 km floor, row-standardised) with the 95% range from 4,999 random permutations of institution labels (grey band); open markers: years with fewer than 33 institutions. (b) Correlogram for 2013–19 using binary distance-band weights (km; 1k = 1,000 km); the number of institution pairs per band is given in parentheses; grey band: 95% permutation range (4,999 permutations). (c) Absolute outcome gap between institution pairs against distance, same-city pairs highlighted; Spearman correlation with permutation p-value (4,999 permutations). The outcome is the period-average deviation of log(1 + mean citing patents) from the year mean. Coordinates: Supplementary Material.
Figure 7 (d). (d) Moran's I of z under distance-based and co-authorship-based weights; bars give the 95% permutation range (49,999 permutations). Ties are measured among works cited by patents only and describe association, not knowledge flows.
"""
(BASE / "captions_revised.txt").write_text(cap, encoding="utf-8"); print("\n캡션 저장:", BASE / "captions_revised.txt"); print("그림 저장:", FIG)

# ---- cell 15 ----
#

# ---- cell 16 ----


# ---- cell 17 ----

