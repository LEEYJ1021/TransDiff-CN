# ===== 01_numbers_v7.py : generated from a notebook; code cells only, outputs removed =====

# ---- cell 00 ----
import io, re, json, zipfile, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 90)

BASE = Path("~/Research/TransDiff-CN_R1").expanduser()
V7, V8 = BASE / "outputs_R1_v7", BASE / "outputs_R1_v8"
OUT = BASE / "outputs_R1_v9_check"; OUT.mkdir(exist_ok=True)
SEED, EVENT = 20261122, 2013
RNG = np.random.default_rng(SEED)

CHECK = []
def chk(item, computed, draft, tol=0.0015, note=""):
    """초안 값(draft)과 재계산 값(computed)을 비교해 CHECK에 쌓음"""
    try:
        ok = (str(computed) == draft) if isinstance(draft, str) else abs(float(computed) - float(draft)) <= tol
        comp = computed if isinstance(computed, str) else round(float(computed), 4)
    except Exception:
        ok, comp = False, computed
    CHECK.append(dict(item=item, computed=comp, draft=draft, status="OK" if ok else "DIFF", note=note))

def norm_inst(s): return re.sub(r"\s*\([^)]*\)\s*$", "", str(s)).strip()

# ---- cell 01 ----
# ---------- 모집단 ----------
pq = BASE / "data_R1" / "papers_unique.parquet"
U = pd.read_parquet(pq) if pq.exists() else pd.read_csv(BASE / "data_R1" / "papers_unique.csv.gz", low_memory=False)
U.columns = U.columns.str.strip()
U["year"] = pd.to_numeric(U["Publication Year"], errors="coerce"); U["cit"] = pd.to_numeric(U["Citing Patents Count"], errors="coerce")
U = U.dropna(subset=["year", "cit"]).copy(); U["year"] = U.year.astype(int)
U = U[U.year.between(2007, 2023)].reset_index(drop=True)
chk("population works", len(U), 213328, tol=0)
chk("cited works (>=1 patent)", int((U.cit >= 1).sum()), 16170, tol=0)
chk("cited share", (U.cit >= 1).mean(), 0.076, tol=0.0005)
print("모집단 연도별 편수 (Lens 쿼리 결과 건수와 대조):\n", U.groupby("year").size().to_string())

# ---------- 옛 33개 파일 ----------
raw = (BASE / "Transport_CN_Scholarly_Works.csv").read_bytes().decode("utf-8", errors="replace")
old = pd.read_csv(io.StringIO(raw), dtype=str, low_memory=False); old.columns = old.columns.str.strip(); del raw
old = old[old["Lens ID"].notna()].copy()
old["year"] = pd.to_numeric(old["Publication Year"], errors="coerce"); old["pat"] = pd.to_numeric(old["Citing Patents Count"], errors="coerce")
old = old[old.year.between(2007, 2023) & old.pat.notna()].copy(); old["year"] = old.year.astype(int)
old = old.drop_duplicates(["Lens ID", "Institution"]); old["inst"] = old.Institution.map(norm_inst)
multi = old.groupby("Lens ID").inst.transform("nunique") > 1
chk("33-inst rows", len(old), 11271, tol=0); chk("33-inst unique works", old["Lens ID"].nunique(), 9294, tol=0)
chk("n institutions", old.inst.nunique(), 33, tol=0); chk("share of rows in multi-institution works", multi.mean(), 0.322, tol=0.001)

# ---------- 후보 기관, 쿼리 파일 ----------
cand = pd.read_csv(BASE / "candidate_institutions_filled.csv", encoding="utf-8-sig")["name"].dropna().astype(str).str.strip().drop_duplicates()
chk("candidate institutions", len(cand), 89, tol=0)
qf = BASE / "extraction_plan" / "export_all_by_year_queries.txt"
if qf.exists():
    qt = qf.read_text(encoding="utf-8", errors="replace")
    print(f"쿼리 파일에서 발견된 후보 기관명: {sum(c in qt for c in cand)}/{len(cand)}")
    i = qt.find("2007"); print("쿼리 파일 2007 부분 앞 500자:\n", qt[max(i - 50, 0): i + 450])
else:
    print("쿼리 파일 없음:", qf)

# ---------- 528 vs 541 셀 ----------
P = pd.read_csv(BASE / "panel_data_v6.csv", encoding="utf-8-sig")
c_csv = old.groupby(["inst", "year"]).size().rename("n_csv").rename_axis(["inst", "year"])
c_rep = P[P.n_papers > 0].assign(inst=lambda d: d.institution.map(norm_inst)).set_index(["inst", "year"]).n_papers.rename("n_repo")
J = pd.concat([c_csv, c_rep], axis=1)
both, only_csv, only_rep = J.n_csv.notna() & J.n_repo.notna(), J.n_csv.notna() & J.n_repo.isna(), J.n_csv.isna() & J.n_repo.notna()
chk("cells, CSV reconstruction", int(J.n_csv.notna().sum()), 541, tol=0); chk("cells, repository (n_papers>0)", int(J.n_repo.notna().sum()), 528, tol=0)
print(f"\n저장소 패널 행 {len(P)} | 공통 {int(both.sum())} | CSV에만 {int(only_csv.sum())} | 저장소에만 {int(only_rep.sum())} | 공통 셀 중 n 불일치 {int((J[both].n_csv != J[both].n_repo).sum())}")
print("CSV에만 있는 셀의 기관별 개수:\n", J[only_csv].groupby(level="inst").size().sort_values(ascending=False).head(10).to_string())
print("※ 'CSV에만' 셀이 528→541 차이의 원인. 본문에는 하나의 정의만 쓰고 설명 한 줄을 추가")

# ---------- 교통 태그: 두 정의를 같은 표본에서 비교 ----------
TR_BROAD = json.loads((V8 / "analysis_manifest_v8.json").read_text())["tr_broad"]
TERMS = (r"\b(?:transport\w*|traffic|railway\w*|railroad\w*|high-speed rail|highway\w*|pavement\w*|vehicle\w*|automotive|automobile\w*|"
         r"logistics|aviation|aircraft|airport\w*|shipping|maritime|public transit|urban transit|subway)\b")
EXCL = r"\b(?:charge|electron|ion|ionic|mass|heat|proton|carrier|gas|water|oxygen|spin|thermal|drug|lipid|membrane)\s+transport\w*"
low = lambda d, c: d[c].fillna("").astype(str).str.lower() if c in d.columns else pd.Series("", index=d.index)
tag_fos = lambda d: low(d, "Fields of Study").str.contains(TR_BROAD, regex=True)
tag_blob = lambda d: (low(d, "Fields of Study") + " " + low(d, "Title") + " " + low(d, "Keywords")).str.replace(EXCL, " ", regex=True).str.contains(TERMS, regex=True)
Uo = U[U["Lens ID"].astype(str).isin(set(old["Lens ID"].astype(str)))]
tg = pd.DataFrame({"sample": ["population", "old33 unique works", "old33 institution-work rows"],
                   "n": [len(U), len(Uo), len(old)],
                   "fos / TR_BROAD": [tag_fos(U).mean(), tag_fos(Uo).mean(), tag_fos(old).mean()],
                   "title+kw+fos / TERMS": [tag_blob(U).mean(), tag_blob(Uo).mean(), tag_blob(old).mean()]})
print("\n교통 태그 비중 (정의별):"); print(tg.round(3).to_string(index=False))
chk("tag share, old33 rows (TERMS)", tg.iloc[2, 4 - 1 + 0] if False else tg.loc[2, "title+kw+fos / TERMS"], 0.292, tol=0.0015, note="초안 29.2%의 정의")
for y, dv in ((2007, 0.426), (2023, 0.187)):
    chk(f"tag share, population {y} (TR_BROAD fos)", tag_fos(U[U.year == y]).mean(), dv, tol=0.0015)
print("※ 29.2%와 42.6%/18.7%가 서로 다른 정의라면 §3.1에 하나로 통일하거나 정의를 각각 명시")

# ---------- 날짜 결측 ----------
dc = [c for c in U.columns if re.search(r"date", c, re.I)]
print("\n모집단의 날짜 열:", dc)
for c in dc:
    print(f"  {c}: 결측 전체 {U[c].isna().mean():.3f} | 인용 논문 {U.loc[U.cit >= 1, c].isna().mean():.3f}")
print("※ 41%는 인용 논문 파일(16,170편)의 값. 한계 절에서 '모집단'이라고 쓰지 말 것")

# ---- cell 02 ----
mf7 = json.loads((V7 / "analysis_manifest_v7.json").read_text()); mf8 = json.loads((V8 / "analysis_manifest_v8.json").read_text())
for k in ("bri_keywords", "ambiguous", "nev_rx", "placebo_rx", "tr_broad"):
    same = mf7.get(k) == mf8.get(k)
    chk(f"manifest v7 == v8: {k}", "same" if same else "different", "same")

# Table A5용 사전 덤프
rows = [(k, "BRI_full", v, k in mf8["ambiguous"]) for k, v in mf8["bri_keywords"].items()]
rows += [("NEV", "NEV", mf8["nev_rx"], False), ("road_safety", "Placebo", mf8["placebo_rx"], False), ("transport tag", "TR_BROAD", mf8["tr_broad"], False)]
pd.DataFrame(rows, columns=["term", "group", "regex", "ambiguous(excluded from BRI_narrow)"]).to_csv(OUT / "table_A5_dictionaries.csv", index=False)

# 사전 문자열이 처음 등장한 파일과 위약 결과 파일의 수정시각
import datetime as dt
key_a, key_b = "high-speed rail", "corridor"
hit = []
for p in BASE.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".py", ".ipynb", ".json", ".txt", ".md"} and p.stat().st_size < 5e6:
        try: t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception: continue
        if key_a in t and key_b in t: hit.append((dt.datetime.fromtimestamp(p.stat().st_mtime), p.relative_to(BASE)))
hit.sort(); print("\n사전 문자열이 들어 있는 파일(수정시각 오름차순, 상위 8):"); [print(" ", a.strftime("%Y-%m-%d %H:%M"), b) for a, b in hit[:8]]
pl = [(dt.datetime.fromtimestamp(p.stat().st_mtime), p.relative_to(BASE)) for p in BASE.rglob("*") if p.is_file() and re.search(r"placebo|verdict|main_table", p.name, re.I) and p.suffix in {".csv", ".png"}]
pl.sort(); print("가장 이른 위약 결과 파일:", pl[:3])
print("※ 수정시각은 복사·재저장으로 바뀔 수 있어 '사전 확정'의 증거는 못 됨. 입증이 안 되면 응답서·본문 문구를 'dictionary reported in Appendix X'로 낮추세요")

# ---- cell 03 ----
# ---------- Table 1 ----------
c03 = pd.read_csv(V8 / "v8_03_compression_contrast.csv", index_col=0)
con = lambda o, late, s: c03[(c03.outcome == o) & (c03.late == late) & (c03.stat == s)].iloc[0]
T1 = {("cum_any", "sd_log"): (-0.135, -0.189, -0.080), ("cum_any", "share_eq1"): (0.064, 0.033, 0.095),
      ("any5_pri", "sd_log"): (-0.107, -0.170, -0.038), ("any5_pri", "share_eq1"): (0.041, 0.003, 0.077),
      ("nonCN5_pri", "sd_log"): (-0.089, -0.163, -0.018), ("nonCN5_pri", "share_eq1"): (0.022, -0.020, 0.063)}
for (o, s), (d, lo, hi) in T1.items():
    r = con(o, "2013-2018", s)
    for nm, cv, dv in (("diff", r["diff"], d), ("lo", r.lo, lo), ("hi", r.hi, hi)): chk(f"T1 {o} {s} {nm}", cv, dv, tol=0.0015)

# ---------- Table 2 ----------
m = pd.read_csv(V8 / "v8_05_main_table_for_manuscript.csv")
T2 = {("cum_any", "all"): (0.586, .263, -0.578, 0.531, 47, 595), ("any5_pri", "all"): (0.521, .900, -0.596, 0.662, 20, 134),
      ("any5_pri", "T_broad"): (0.333, .988, -0.661, 0.610, 11, 89), ("any3_pri", "all"): (0.547, .797, -0.663, 0.743, 15, 234)}
PRIM = "BRI_full[txt]"
for (o, pop), vals in T2.items():
    r = m[(m.outcome == o) & (m.population == pop) & (m.target == PRIM)].iloc[0]
    for nm, cv, dv in zip(("DID", "p_adj", "q025", "q975", "cited_pre", "cited_post"),
                          (r.observed_DID_long, r.p_adj_DID_long, r.q025_DID_long, r.q975_DID_long, r.cited_pre, r.cited_post), vals):
        chk(f"T2 {o}/{pop} {nm}", cv, dv, tol=0.0015 if nm not in ("cited_pre", "cited_post") else 0)

# ---------- 구간 문구: '5-20편', '47편', '±0.6-0.7' ----------
bri = m[m.target.str.startswith("BRI")]; fixed = ["any3_pri", "any5_pri", "nonCN5_pri"]
def rng_(d, c): return f"{int(d[c].min())}–{int(d[c].max())}"
print("\n[cited_pre 범위] 초안은 '5–20(고정 창), 47(누적)'")
print("  주 사양(BRI_full[txt]) 고정 창:", rng_(m[(m.target == PRIM) & m.outcome.isin(fixed)], "cited_pre"))
print("  모든 BRI 정의 고정 창      :", rng_(bri[bri.outcome.isin(fixed)], "cited_pre"))
print("  모든 BRI 정의 고정 창, nonCN 제외:", rng_(bri[bri.outcome.isin(["any3_pri", "any5_pri"])], "cited_pre"))
print("  주 사양 누적              :", rng_(m[(m.target == PRIM) & (m.outcome == "cum_any")], "cited_pre"), "| 모든 BRI 정의 누적:", rng_(bri[bri.outcome == "cum_any"], "cited_pre"))
for o in fixed: print(f"  귀무 95% 범위 {o}: [{bri[bri.outcome == o].q025_DID_long.min():.2f}, {bri[bri.outcome == o].q975_DID_long.max():.2f}]")
print("※ '5–20'이 어느 집합(주 사양/전체 BRI 정의)의 범위인지 본문에 명시. 범위가 다르면 문장 수정")

# ---------- 코호트 수치(§4.1) ----------
c01 = pd.read_csv(V8 / "v8_01_cohort_compression.csv")
cv = lambda o, y, c: c01[(c01.outcome == o) & (c01.year == y)][c].iloc[0]
chk("cum cited share 2017", cv("cum_any", 2017, "cited_share"), 0.115); chk("cum cited share 2023", cv("cum_any", 2023, "cited_share"), 0.055)
chk("5y cited share 2017", cv("any5_pri", 2017, "cited_share"), 0.10, tol=0.003); chk("5y cited share 2018", cv("any5_pri", 2018, "cited_share"), 0.10, tol=0.003)
chk("cum share_eq1 2007", cv("cum_any", 2007, "share_eq1"), 0.47, tol=0.005); chk("cum share_eq1 2023", cv("cum_any", 2023, "share_eq1"), 0.83, tol=0.005)
for o in ("any5_pri", "any3_pri"):
    s = c01[c01.outcome == o].share_eq1; print(f"  share_eq1 범위 {o}: {s.min():.3f}–{s.max():.3f} (초안: 0.66–0.72)")
s3 = c01[(c01.outcome == "any3_pri") & c01.year.between(2017, 2020)].cited_share; print(f"  3년 창 cited share 2017–2020: {s3.min():.3f}–{s3.max():.3f} (초안 'about 8%')")

# ---------- 링크 품질, 패밀리 구성 ----------
qa = pd.read_csv(V7 / "00_link_QA.csv", index_col=0)["v"]
chk("link coverage", qa["link coverage"], 0.998, tol=0.0006); chk("exact match share", qa["exact match share"], 0.996, tol=0.0006); chk("corr(linked, Lens)", qa["corr(linked_docs, Lens count)"], 0.9997, tol=0.0002)
comp = pd.read_csv(V7 / "06_citing_family_composition.csv"); comp = comp[comp.k == 5].set_index("py")
chk("CN-only share 2007", comp.loc[2007, "share_CN_only"], 0.07, tol=0.005); chk("CN-only share 2018", comp.loc[2018, "share_CN_only"], 0.37, tol=0.005)

# ---------- 33개 표본의 대표성(coverage) ----------
Uc = U[U.cit >= 1].copy(); Uc["in_old"] = Uc["Lens ID"].astype(str).isin(set(old["Lens ID"].astype(str)))
cov = Uc.groupby("year").in_old.mean(); chk("old33 coverage of cited 2007", cov[2007], 0.92, tol=0.006); chk("old33 coverage of cited 2023", cov[2023], 0.35, tol=0.006)

# ---------- σ, 공동논문 제외, ICC, 셀 크기 고정 ----------
def sig(y_ser): return y_ser.groupby(level="year").std(ddof=1)
def pre_end(s): return s.loc[2007:2012].mean(), s.loc[2021:2023].mean()
y_s0 = np.log1p(old.groupby(["inst", "year"]).pat.mean()); s0 = sig(y_s0); a, b = pre_end(s0)
chk("σ pre (CSV, S0)", a, 0.480); chk("σ end (CSV, S0)", b, 0.107); chk("σ decline % (S0)", 100 * (b / a - 1), -77.8, tol=0.15)
Pr = P[P.n_papers > 0].assign(y=lambda d: np.log1p(d.mean_patent_cit), inst=lambda d: d.institution.map(norm_inst)).set_index(["inst", "year"]).y
a2, b2 = pre_end(sig(Pr)); chk("σ pre (repository, empty cells excluded)", a2, 0.478); chk("σ end (repository, empty cells excluded)", b2, 0.109)
def s_decline(d, kind):
    y = np.log1p(d.groupby(["inst", "year"]).pat.mean()) if kind == "S0" else np.log1p(d.pat).groupby([d.inst, d.year]).mean()
    p_, e_ = pre_end(sig(y)); return 100 * (e_ / p_ - 1)
nsh = old[~multi]; print(f"\n공동 논문 제외 σ 감소율: S0 {s_decline(nsh, 'S0'):.1f}% | S1 {s_decline(nsh, 'S1'):.1f}%")
chk("σ decline excl. shared (S1)", s_decline(nsh, "S1"), -75.7, tol=0.15, note="초안은 S0 설명 직후에 이 값을 씀: S1 기준임을 명시하거나 S0 값으로 교체")

M = y_s0.unstack("year"); srat = lambda X: X.std(axis=0, ddof=1).loc[2021:2023].mean() / X.std(axis=0, ddof=1).loc[2015:2017].mean()
bs = np.array([srat(M.iloc[RNG.integers(0, len(M), len(M))]) for _ in range(2000)])
chk("σ ratio 2021-23 / 2015-17", srat(M), 0.489, tol=0.005); chk("σ ratio CI lo", np.percentile(bs, 2.5), 0.323, tol=0.03); chk("σ ratio CI hi", np.percentile(bs, 97.5), 0.679, tol=0.03)

def icc1(g):
    g = g[g.groupby("inst").v.transform("size") >= 2]; grp = g.groupby("inst").v; ni, mi = grp.size().values, grp.mean().values
    N, k, gm = ni.sum(), len(ni), g.v.mean()
    if k < 5: return np.nan
    msb = (ni * (mi - gm) ** 2).sum() / (k - 1); msw = ((g.v - grp.transform("mean")) ** 2).sum() / (N - k); n0 = (N - (ni ** 2).sum() / N) / (k - 1)
    return max((msb - msw) / (msb + (n0 - 1) * msw), 0)
d_ = old.assign(v=np.log1p(old.pat))
vd = pd.DataFrame([(t, g.v.std(ddof=1), (g.pat >= 3).mean(), icc1(g)) for t, g in d_.groupby("year")], columns=["year", "paper_sd", "share3", "icc"]).set_index("year")
chk("paper SD 2007", vd.paper_sd[2007], 0.650); chk("paper SD 2023", vd.paper_sd[2023], 0.225)
chk("paper SD decline %", 100 * (vd.paper_sd[2023] / vd.paper_sd[2007] - 1), -65, tol=1.0); chk("share>=3 2007", vd.share3[2007], 0.38, tol=0.005); chk("share>=3 2023", vd.share3[2023], 0.03, tol=0.005)
chk("ICC max over years", vd.icc.max(), 0.10, tol=0.005)

def sigma_matched(d, m_=5, reps=300, min_inst=10):
    d = d.assign(v=np.log1p(d.pat)); d = d[d.groupby(["inst", "year"]).v.transform("size") >= m_]
    ni = d.groupby("year").inst.nunique(); d = d[d.year.isin(ni[ni >= min_inst].index)]
    S = pd.concat([d.groupby(["inst", "year"]).sample(n=m_, random_state=SEED + r).groupby(["inst", "year"]).v.mean().groupby("year").std(ddof=1) for r in range(reps)], axis=1)
    return pd.DataFrame({"sigma": S.mean(1), "n_inst": ni.reindex(S.index)})
sm = sigma_matched(old, 5); chk("matched σ 2017 (m=5)", sm.sigma[2017], 0.224, tol=0.004); chk("matched σ 2023 (m=5)", sm.sigma[2023], 0.104, tol=0.004)
chk("n_inst 2017", sm.n_inst[2017], 33, tol=0); chk("n_inst 2023", sm.n_inst[2023], 33, tol=0)

# ---- cell 04 ----
need = ["df", "META", "names", "make_panel", "moran_perm", "W_inv", "W_inv_xcity", "W_knn", "dist_matrix", "rowstd"]
miss = [n for n in need if n not in globals()]
if miss:
    print("노트북에서 해당 정의 셀을 먼저 실행하세요:", miss)
else:
    d2 = df.assign(Institution=df.Institution.map(norm_inst))
    PIV = make_panel(d2, "log_mean").pivot(index="Institution", columns="year", values="y").reindex(names)
    Prp = P.assign(y=np.log1p(P.mean_patent_cit.fillna(0)))
    D0 = dist_matrix([META[n][1] for n in names], [META[n][2] for n in names])
    city = np.array([META[n][0] for n in names]); typ = np.array([META[n][4] for n in names])
    Wrepo = {k: np.load(BASE / f"W_{k}.npy") for k in ("geo", "type", "region", "combo")}
    Y0_, Y1_ = int(PIV.columns.min()), int(PIV.columns.max())
    PERS = {"2013-19": (EVENT, 2019), f"{Y0_}-{Y1_}": (Y0_, Y1_)}
    ydef = {"mine": lambda a, b: PIV.loc[:, a:b].mean(axis=1).values,
            "repo": lambda a, b: Prp[Prp.year.between(a, b)].groupby("institution").y.mean().reindex(names).values}

    def run(y, W, mask=None):
        k = ~np.isnan(y) if mask is None else (~np.isnan(y)) & mask
        Wk = W(D0[np.ix_(k, k)]) if callable(W) else rowstd(W[np.ix_(k, k)])
        I, p, _ = moran_perm(y[k], Wk); return I, p, int(k.sum())

    geoW = {"inv no floor": lambda d: W_inv(d, floor=1e-6), "inv floor25": lambda d: W_inv(d, floor=25.), "inv floor100": lambda d: W_inv(d, floor=100.),
            "excl same-city": W_inv_xcity, "kNN3": lambda d: W_knn(d, 3), "kNN5": lambda d: W_knn(d, 5)}
    rows = []
    for yn, yf in ydef.items():
        for pn, (a, b) in PERS.items():
            y = yf(a, b)
            for sn, W in geoW.items(): rows.append((yn, pn, sn, *run(y, W)))
            rows.append((yn, pn, "excl Beijing", *run(y, geoW["inv floor25"], mask=(city != "Beijing"))))
            rows.append((yn, pn, "Beijing only", *run(y, geoW["inv floor25"], mask=(city == "Beijing"))))
            for wn, Wm in Wrepo.items(): rows.append((yn, pn, f"repo W_{wn}", *run(y, Wm)))
            cu = pd.unique(city); yc = np.array([np.nanmean(y[city == c]) for c in cu])
            la = np.array([np.mean([META[n][1] for n in names if META[n][0] == c]) for c in cu]); lo = np.array([np.mean([META[n][2] for n in names if META[n][0] == c]) for c in cu])
            Ic, pc, _ = moran_perm(yc, W_inv(dist_matrix(la, lo), floor=25.)); rows.append((yn, pn, "city-level floor25", Ic, pc, len(cu)))
    M3 = pd.DataFrame(rows, columns=["y_def", "period", "spec", "I", "p", "n"]); M3.to_csv(OUT / "table3_moran_all_periods.csv", index=False)
    print(M3.pivot_table(index=["spec"], columns=["y_def", "period"], values="I").round(3).to_string())

    full = f"{Y0_}-{Y1_}"
    T3 = [("no floor, reconstructed y", "mine", full, "inv no floor", -0.317, .010), ("no floor, repository y", "repo", full, "inv no floor", -0.120, .293),
          ("floor25", "mine", "2013-19", "inv floor25", -0.152, .149), ("floor100", "mine", "2013-19", "inv floor100", -0.083, .297),
          ("excl same-city", "mine", "2013-19", "excl same-city", 0.002, .479), ("kNN3", "mine", "2013-19", "kNN3", -0.207, .162), ("kNN5", "mine", "2013-19", "kNN5", -0.113, .357),
          ("excl Beijing", "mine", "2013-19", "excl Beijing", -0.183, .203), ("city-level", "mine", "2013-19", "city-level floor25", -0.095, .626),
          ("composite", "repo", full, "repo W_combo", 0.015, .454), ("type only", "repo", full, "repo W_type", 0.323, .001)]
    for lab, yn, pn, sn, dI, dp in T3:
        r = M3[(M3.y_def == yn) & (M3.period == pn) & (M3.spec == sn)].iloc[0]
        chk(f"T3 {lab} I", r.I, dI, tol=0.003, note=f"y={yn}, period={pn}"); chk(f"T3 {lab} p", r.p, dp, tol=0.02, note="순열 p는 시드·횟수에 따라 ±0.01")
    print("※ Table 3의 각 행에 기간(y 정의 포함)을 표기하거나, 한 기간으로 통일해 다시 채우세요")

    # ---- 유형 η² (원 분석의 유형 라벨 사용) ----
    def eta(v, t): gm = v.mean(); return sum((t == g).sum() * (v[t == g].mean() - gm) ** 2 for g in np.unique(t)) / ((v - gm) ** 2).sum()
    for (lab, (a, b)), dI, dp in zip({"2007-12": (2007, 2012), "2013-19": (2013, 2019), "2020-23": (2020, 2023)}.items(), (0.331, 0.283, 0.207), (.020, .040, .149)):
        v = PIV.loc[:, a:b].mean(axis=1).values; k = ~np.isnan(v); e = eta(v[k], typ[k])
        pp = np.mean([eta(RNG.permutation(v[k]), typ[k]) >= e for _ in range(4999)])
        chk(f"η² type {lab}", e, dI, tol=0.003); chk(f"η² perm p {lab}", pp, dp, tol=0.02)

    # ---- 관문 근접도: 목록과 민감도 ----
    need2 = ["geo", "gw", "piv0", "hav", "ORDER"]
    if [n for n in need2 if n not in globals()]:
        print("관문 민감도: Cell 2~4 객체(geo, gw, piv0, hav, ORDER)가 필요")
    else:
        print("\n관문 목록(좌표를 직접 검증하세요):\n", gw[["name", "lat", "lon"]].to_string(index=False))
        P0 = piv0.reindex(ORDER); ya = int(piv0.columns.min())
        dlt = P0.loc[:, EVENT:2019].mean(axis=1).values - P0.loc[:, ya:EVENT - 1].mean(axis=1).values
        g_ = geo.set_index("Institution").loc[ORDER]
        def grad(gwd, B=4999):
            gk = np.array([hav(r.lat, r.lon, gwd.lat.values, gwd.lon.values).min() for r in g_.itertuples()])
            pr = stats.zscore(-np.log(gk)); k = ~np.isnan(dlt); sl = stats.linregress(pr[k], dlt[k]).slope
            X = pr[k][np.argsort(RNG.random((B, k.sum())), axis=1)]; Xc = X - X.mean(1, keepdims=True)
            sims = (Xc @ (dlt[k] - dlt[k].mean())) / (Xc ** 2).sum(1)
            return sl, (1 + (np.abs(sims) >= abs(sl)).sum()) / (B + 1)
        port = gw.name.str.contains(r"Ningbo|Yangshan|Yantian|Nansha|Qingdao|port|Xiamen", case=False)
        var = {"all": gw, "land only": gw[~port], "ports only": gw[port], **{f"drop {n}": gw[gw.name != n] for n in gw.name}}
        GW = pd.DataFrame([(k, *grad(v), len(v)) for k, v in var.items()], columns=["variant", "slope", "perm_p", "n_gateways"])
        print("\n관문 목록 민감도:\n", GW.round(3).to_string(index=False)); GW.to_csv(OUT / "gateway_sensitivity.csv", index=False)
        chk("gateway slope (all)", GW.slope[0], -0.027, tol=0.004); chk("gateway perm p (all)", GW.perm_p[0], 0.35, tol=0.03)
        print("※ 모든 변형에서 p가 크고 부호가 같으면 '목록에 의존하지 않는 null'로 쓸 수 있음. 목록 검증이 안 되면 해당 문장 삭제")

# ---- cell 05 ----
SUBMIT_DIRS = [BASE / "repo_snapshot"]          # 새 익명 저장소 폴더와 보충자료 폴더를 추가하세요
PATTERNS = ["LEEYJ1021", "yj11021", "hanyang", "Yong-Jae", "Yongjae", ".", "yjlee", "Sangjin", "tobesoft", "korea.ac.kr"]  # 본인 식별 문자열
TEXT_EXT = {".py", ".ipynb", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".tex", ".html", ".bib", ".sh", ".ini"}
hits = []
for root in SUBMIT_DIRS:
    if not root.exists(): print("폴더 없음(건너뜀):", root); continue
    for p in root.rglob("*"):
        if not p.is_file(): continue
        sfx = p.suffix.lower()
        if p.name.lower().startswith(".git") or ".git" in p.parts: hits.append((str(p), "git metadata", ".git 폴더 포함(커밋 작성자 노출 가능)")); continue
        if sfx in TEXT_EXT and p.stat().st_size < 2e7:
            t = p.read_text(encoding="utf-8", errors="ignore")
            for pat in PATTERNS:
                n = len(re.findall(re.escape(pat), t, re.I))
                if n: hits.append((str(p), pat, f"{n}회: ..." + t[max(re.search(re.escape(pat), t, re.I).start() - 30, 0): re.search(re.escape(pat), t, re.I).start() + 50].replace("\n", " ") + "..."))
        if sfx in {".docx", ".xlsx", ".pptx"}:
            with zipfile.ZipFile(p) as z:
                for part in ("docProps/core.xml", "docProps/app.xml"):
                    if part in z.namelist():
                        x = z.read(part).decode("utf-8", errors="ignore")
                        for tag in ("dc:creator", "cp:lastModifiedBy", "Company", "Manager"):
                            mm = re.search(rf"<{tag}>([^<]*)</{tag}>", x)
                            if mm and mm.group(1).strip(): hits.append((str(p), f"metadata {tag}", mm.group(1)))
        if sfx == ".pdf":
            try:
                from pypdf import PdfReader
                md = PdfReader(str(p)).metadata or {}
                for k in ("/Author", "/Creator", "/Producer", "/Title"):
                    if md.get(k): hits.append((str(p), f"pdf {k}", str(md.get(k))))
            except Exception as e: hits.append((str(p), "pdf metadata", f"확인 불가({e}); 수동 확인"))
H = pd.DataFrame(hits, columns=["file", "pattern/field", "detail"]); H.to_csv(OUT / "anonymity_scan.csv", index=False)
print("\n[익명성 스캔]", "문제 없음(패턴 기준)" if H.empty else f"{len(H)}건"); print(H.to_string(index=False) if len(H) else "")
print("※ ipynb 출력에는 홈 경로·사용자명이 남기 쉬움. 링크가 로그인 없이 열리는지는 시크릿 창에서 직접 확인")

# ---------- 최종 대조표 ----------
C = pd.DataFrame(CHECK); C.to_csv(OUT / "check_summary.csv", index=False, encoding="utf-8-sig")
print(f"\n=== 초안 대조 결과: OK {int((C.status == 'OK').sum())} / DIFF {int((C.status == 'DIFF').sum())} ===")
print(C.sort_values("status", ascending=False).to_string(index=False))

# ---- cell 06 ----
#

# ---- cell 07 ----
import io, re, numpy as np, pandas as pd
from pathlib import Path
BASE = Path("~/Research/TransDiff-CN_R1").expanduser(); OUT = BASE / "outputs_R1_v9_check"; OUT.mkdir(exist_ok=True)
SEED, EVENT = 20261122, 2013; RNG = np.random.default_rng(SEED)
norm = lambda s: re.sub(r"\s*\([^)]*\)\s*$", "", str(s)).strip()

META = {  # name: (city, lat, lon, region, type)  (step1 INST_META와 동일)
 "Chinese Academy of Sciences":("Beijing",39.9042,116.4074,"North","research"),"Tsinghua University":("Beijing",40.0033,116.3267,"North","C9"),
 "Beijing Jiaotong University":("Beijing",39.9647,116.3539,"North","transport"),"Tongji University":("Shanghai",31.2813,121.5034,"East","transport"),
 "Southeast University":("Nanjing",32.1578,118.8208,"East","C9"),"Zhejiang University":("Hangzhou",30.2587,120.1310,"East","C9"),
 "Shanghai Jiao Tong University":("Shanghai",31.2024,121.4337,"East","C9"),"Beihang University":("Beijing",40.0009,116.3428,"North","C7"),
 "Wuhan University of Technology":("Wuhan",30.5099,114.3974,"Central","transport"),"Central South University":("Changsha",28.1804,112.9321,"Central","transport"),
 "Harbin Institute of Technology":("Harbin",45.7761,126.6723,"Northeast","C9"),"University of Chinese Academy of Sciences":("Beijing",39.9975,116.3862,"North","research"),
 "Peking University":("Beijing",39.9971,116.3107,"North","C9"),"Jilin University":("Changchun",43.8878,125.3247,"Northeast","C9"),
 "Wuhan University":("Wuhan",30.5389,114.3599,"Central","C9"),"South China University of Technology":("Guangzhou",23.0490,113.4022,"South","C9"),
 "Shandong University":("Jinan",36.6804,117.0601,"North","C9"),"Sun Yat-sen University":("Guangzhou",23.0932,113.2978,"South","C9"),
 "Huazhong University of Science and Technology":("Wuhan",30.5091,114.4139,"Central","C9"),"Beijing Institute of Technology":("Beijing",39.9652,116.3223,"North","C7"),
 "Shenzhen University":("Shenzhen",22.5341,113.9298,"South","teaching"),"Southwest Jiaotong University":("Chengdu",30.6441,103.8418,"West","transport"),
 "Fudan University":("Shanghai",31.2985,121.5018,"East","C9"),"Sichuan University":("Chengdu",30.6310,104.0821,"West","C9"),
 "Tianjin University":("Tianjin",39.1053,117.1619,"North","C9"),"Dalian University of Technology":("Dalian",38.8574,121.5264,"Northeast","C9"),
 "Chongqing University":("Chongqing",29.5594,106.5509,"West","C9"),"Nanjing University":("Nanjing",32.1146,118.7835,"East","C9"),
 "Soochow University":("Suzhou",31.3019,120.6239,"East","teaching"),"Xi'an Jiaotong University":("Xi'an",34.2570,108.9784,"West","C9"),
 "University of Science and Technology of China":("Hefei",31.8315,117.2571,"East","C9"),"Shanghai University":("Shanghai",31.2835,121.4440,"East","teaching"),
 "Beijing University of Technology":("Beijing",39.8590,116.4815,"North","teaching")}
names = list(META)

# ---- 패널(CSV 재구성): 기관-연도 log(mean+1) ----
raw = (BASE / "Transport_CN_Scholarly_Works.csv").read_bytes().decode("utf-8", errors="replace")
old = pd.read_csv(io.StringIO(raw), dtype=str, low_memory=False); old.columns = old.columns.str.strip(); del raw
old = old[old["Lens ID"].notna()].copy()
old["year"] = pd.to_numeric(old["Publication Year"], errors="coerce"); old["pat"] = pd.to_numeric(old["Citing Patents Count"], errors="coerce")
old = old[old.year.between(2007, 2023) & old.pat.notna()].copy(); old["year"] = old.year.astype(int)
old["inst"] = old.Institution.map(norm); old = old.drop_duplicates(["Lens ID", "inst"])
PIV = np.log1p(old.groupby(["inst", "year"]).pat.mean()).unstack("year").reindex(names)
P = pd.read_csv(BASE / "panel_data_v6.csv", encoding="utf-8-sig"); P["y"] = np.log1p(P.mean_patent_cit.fillna(0))

# ---- 거리, W, Moran ----
def hav(la1, lo1, la2, lo2):
    p = np.pi/180; a = np.sin((la2-la1)*p/2)**2 + np.cos(la1*p)*np.cos(la2*p)*np.sin((lo2-lo1)*p/2)**2; return 12742*np.arcsin(np.sqrt(a))
def dmat(la, lo): la, lo = np.asarray(la), np.asarray(lo); return hav(la[:,None], lo[:,None], la[None,:], lo[None,:])
def rs(W): s = W.sum(1, keepdims=True); return np.divide(W, s, out=np.zeros_like(W), where=s > 0)
def Winv(D, floor=1e-6): W = 1/np.maximum(D, floor); np.fill_diagonal(W, 0); return rs(W)
def Wxc(D, floor=25., km=30.): W = 1/np.maximum(D, floor); W[D < km] = 0; np.fill_diagonal(W, 0); return rs(W)
def Wknn(D, k): W = np.zeros_like(D); [W.__setitem__((i, np.argsort(D[i])[1:k+1]), 1) for i in range(len(D))]; return rs(W)
def moran_perm(y, W, B=4999):
    n = len(y); z = y - y.mean(); I = n/W.sum()*(z @ W @ z)/(z @ z)
    Z = z[np.argsort(RNG.random((B, n)), axis=1)]; sims = n/W.sum()*((Z @ W)*Z).sum(1)/(Z**2).sum(1)
    return I, (1 + (np.abs(sims - sims.mean()) >= abs(I - sims.mean())).sum())/(B + 1)
D0 = dmat([META[n][1] for n in names], [META[n][2] for n in names])
city = np.array([META[n][0] for n in names]); typ = np.array([META[n][4] for n in names])
Wrepo = {k: np.load(BASE / f"W_{k}.npy") for k in ("geo", "type", "region", "combo")}

def run(y, W, mask=None):
    k = ~np.isnan(y) if mask is None else (~np.isnan(y)) & mask
    Wk = W(D0[np.ix_(k, k)]) if callable(W) else rs(W[np.ix_(k, k)])
    return (*moran_perm(y[k], Wk), int(k.sum()))

ydef = {"mine": lambda a, b: PIV.loc[:, a:b].mean(axis=1).values,
        "repo": lambda a, b: P[P.year.between(a, b)].groupby("institution").y.mean().reindex(names).values}
geoW = {"inv no floor": Winv, "inv floor25": lambda d: Winv(d, 25.), "inv floor100": lambda d: Winv(d, 100.),
        "excl same-city": Wxc, "kNN3": lambda d: Wknn(d, 3), "kNN5": lambda d: Wknn(d, 5)}
rows = []
for yn, yf in ydef.items():
    for pn, (a, b) in {"2013-19": (2013, 2019), "2007-23": (2007, 2023)}.items():
        y = yf(a, b)
        for sn, W in geoW.items(): rows.append((yn, pn, sn, *run(y, W)))
        rows += [(yn, pn, "excl Beijing", *run(y, geoW["inv floor25"], city != "Beijing")),
                 (yn, pn, "Beijing only", *run(y, geoW["inv floor25"], city == "Beijing"))]
        for wn, Wm in Wrepo.items(): rows.append((yn, pn, f"repo W_{wn}", *run(y, Wm)))
        cu = pd.unique(city); yc = np.array([np.nanmean(y[city == c]) for c in cu])
        la = np.array([np.mean([META[n][1] for n in names if META[n][0] == c]) for c in cu]); lo = np.array([np.mean([META[n][2] for n in names if META[n][0] == c]) for c in cu])
        rows.append((yn, pn, "city-level floor25", *moran_perm(yc, Winv(dmat(la, lo), 25.)), len(cu)))
M3 = pd.DataFrame(rows, columns=["y_def", "period", "spec", "I", "p", "n"]); M3.to_csv(OUT / "table3_moran_all_periods.csv", index=False)
print(M3.pivot_table(index="spec", columns=["y_def", "period"], values="I").round(3).to_string())
print(M3.pivot_table(index="spec", columns=["y_def", "period"], values="p").round(3).to_string())

# ---- 초안 Table 3 대조 ----
T3 = [("no floor, reconstructed y", "mine", "2007-23", "inv no floor", -0.317), ("no floor, repository y", "repo", "2007-23", "inv no floor", -0.120),
      ("floor25", "mine", "2013-19", "inv floor25", -0.152), ("floor100", "mine", "2013-19", "inv floor100", -0.083),
      ("excl same-city", "mine", "2013-19", "excl same-city", 0.002), ("kNN3", "mine", "2013-19", "kNN3", -0.207), ("kNN5", "mine", "2013-19", "kNN5", -0.113),
      ("excl Beijing", "mine", "2013-19", "excl Beijing", -0.183), ("city-level", "mine", "2013-19", "city-level floor25", -0.095),
      ("composite", "repo", "2007-23", "repo W_combo", 0.015), ("type only", "repo", "2007-23", "repo W_type", 0.323)]
for lab, yn, pn, sn, dI in T3:
    r = M3[(M3.y_def == yn) & (M3.period == pn) & (M3.spec == sn)].iloc[0]
    print(f"{lab:28s} 초안 {dI:+.3f} | 재계산 {r.I:+.3f} (p={r.p:.3f}, n={r.n}) [{yn}, {pn}] {'OK' if abs(r.I-dI)<=0.003 else 'DIFF'}")

# ---- 유형 η² ----
eta = lambda v, t: sum((t == g).sum()*(v[t == g].mean()-v.mean())**2 for g in np.unique(t))/((v-v.mean())**2).sum()
for lab, (a, b), dI in (("2007-12", (2007, 2012), 0.331), ("2013-19", (2013, 2019), 0.283), ("2020-23", (2020, 2023), 0.207)):
    v = PIV.loc[:, a:b].mean(axis=1).values; k = ~np.isnan(v); e = eta(v[k], typ[k])
    pp = np.mean([eta(RNG.permutation(v[k]), typ[k]) >= e for _ in range(4999)])
    print(f"η² type {lab}: 초안 {dI:.3f} | 재계산 {e:.3f} (perm p={pp:.3f})")

# ---- cell 08 ----
#

# ---- cell 09 ----
import io, re, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

BASE = Path("~/Research/TransDiff-CN_R1").expanduser(); OUT = BASE / "outputs_R1_v9_check"; OUT.mkdir(exist_ok=True)
SEED, EVENT, B = 20261122, 2013, 4999; RNG = np.random.default_rng(SEED)
norm = lambda s: re.sub(r"\s*\([^)]*\)\s*$", "", str(s)).strip()

META = {  # name: (city, lat, lon, region, type)
 "Chinese Academy of Sciences":("Beijing",39.9042,116.4074,"North","research"),"Tsinghua University":("Beijing",40.0033,116.3267,"North","C9"),
 "Beijing Jiaotong University":("Beijing",39.9647,116.3539,"North","transport"),"Tongji University":("Shanghai",31.2813,121.5034,"East","transport"),
 "Southeast University":("Nanjing",32.1578,118.8208,"East","C9"),"Zhejiang University":("Hangzhou",30.2587,120.1310,"East","C9"),
 "Shanghai Jiao Tong University":("Shanghai",31.2024,121.4337,"East","C9"),"Beihang University":("Beijing",40.0009,116.3428,"North","C7"),
 "Wuhan University of Technology":("Wuhan",30.5099,114.3974,"Central","transport"),"Central South University":("Changsha",28.1804,112.9321,"Central","transport"),
 "Harbin Institute of Technology":("Harbin",45.7761,126.6723,"Northeast","C9"),"University of Chinese Academy of Sciences":("Beijing",39.9975,116.3862,"North","research"),
 "Peking University":("Beijing",39.9971,116.3107,"North","C9"),"Jilin University":("Changchun",43.8878,125.3247,"Northeast","C9"),
 "Wuhan University":("Wuhan",30.5389,114.3599,"Central","C9"),"South China University of Technology":("Guangzhou",23.0490,113.4022,"South","C9"),
 "Shandong University":("Jinan",36.6804,117.0601,"North","C9"),"Sun Yat-sen University":("Guangzhou",23.0932,113.2978,"South","C9"),
 "Huazhong University of Science and Technology":("Wuhan",30.5091,114.4139,"Central","C9"),"Beijing Institute of Technology":("Beijing",39.9652,116.3223,"North","C7"),
 "Shenzhen University":("Shenzhen",22.5341,113.9298,"South","teaching"),"Southwest Jiaotong University":("Chengdu",30.6441,103.8418,"West","transport"),
 "Fudan University":("Shanghai",31.2985,121.5018,"East","C9"),"Sichuan University":("Chengdu",30.6310,104.0821,"West","C9"),
 "Tianjin University":("Tianjin",39.1053,117.1619,"North","C9"),"Dalian University of Technology":("Dalian",38.8574,121.5264,"Northeast","C9"),
 "Chongqing University":("Chongqing",29.5594,106.5509,"West","C9"),"Nanjing University":("Nanjing",32.1146,118.7835,"East","C9"),
 "Soochow University":("Suzhou",31.3019,120.6239,"East","teaching"),"Xi'an Jiaotong University":("Xi'an",34.2570,108.9784,"West","C9"),
 "University of Science and Technology of China":("Hefei",31.8315,117.2571,"East","C9"),"Shanghai University":("Shanghai",31.2835,121.4440,"East","teaching"),
 "Beijing University of Technology":("Beijing",39.8590,116.4815,"North","teaching")}
names = list(META); n = len(names)
city = np.array([META[k][0] for k in names]); typ = np.array([META[k][4] for k in names]); mac = np.array([META[k][3] for k in names])

# ---- 재구성 y (log(mean+1) 기관-연도 평균 -> 기간 평균) ----
raw = (BASE / "Transport_CN_Scholarly_Works.csv").read_bytes().decode("utf-8", errors="replace")
old = pd.read_csv(io.StringIO(raw), dtype=str, low_memory=False); old.columns = old.columns.str.strip(); del raw
old = old[old["Lens ID"].notna()].copy()
old["year"] = pd.to_numeric(old["Publication Year"], errors="coerce"); old["pat"] = pd.to_numeric(old["Citing Patents Count"], errors="coerce")
old = old[old.year.between(2007, 2023) & old.pat.notna()].copy(); old["year"] = old.year.astype(int)
old["inst"] = old.Institution.map(norm); old = old.drop_duplicates(["Lens ID", "inst"])
PIV = np.log1p(old.groupby(["inst", "year"]).pat.mean()).unstack("year").reindex(names)
y = PIV.loc[:, 2013:2019].mean(axis=1).values
print("n 기관:", n, "| y 결측:", int(np.isnan(y).sum()))

def hav(la1, lo1, la2, lo2):
    p = np.pi/180; a = np.sin((la2-la1)*p/2)**2 + np.cos(la1*p)*np.cos(la2*p)*np.sin((lo2-lo1)*p/2)**2; return 12742*np.arcsin(np.sqrt(a))
la = np.array([META[k][1] for k in names]); lo = np.array([META[k][2] for k in names])
D = hav(la[:, None], lo[:, None], la[None, :], lo[None, :])

# =================== 1) 쌍별 격차: Mantel형 + 쌍 회귀 (n=33) ===================
k = ~np.isnan(y); yk = y[k]; Dk = D[np.ix_(k, k)]; iu = np.triu_indices(len(yk), 1)
dist, gap = Dk[iu], np.abs(yk[:, None] - yk[None, :])[iu]
rho = stats.spearmanr(dist, gap)[0]
Pm = np.argsort(RNG.random((B, len(yk))), axis=1); G = np.abs(yk[Pm][:, iu[0]] - yk[Pm][:, iu[1]])
rg = stats.rankdata(G, axis=1); rg -= rg.mean(1, keepdims=True); rd = stats.rankdata(dist); rd -= rd.mean()
sims = (rg @ rd)/np.sqrt((rg**2).sum(1)*(rd**2).sum())
p_m = (1 + (np.abs(sims) >= abs(rho)).sum())/(B + 1)
print(f"\n[Mantel] n={len(yk)} | Spearman ρ(gap, distance) = {rho:.3f} | perm p = {p_m:.3f}   (초안: ρ=-0.02, p=.73 @ n=32)")

ci, cj = city[k][iu[0]], city[k][iu[1]]
Z = np.column_stack([np.ones(len(iu[0])), ci == cj, typ[k][iu[0]] == typ[k][iu[1]], mac[k][iu[0]] == mac[k][iu[1]], np.log(np.maximum(dist, 25))]).astype(float)
coef = lambda v: np.linalg.lstsq(Z, np.abs(v[iu[0]] - v[iu[1]]), rcond=None)[0]
b0 = coef(yk); Bs = np.array([coef(yk[p]) for p in Pm])
rows = [(nm, b0[j], (1 + (np.abs(Bs[:, j] - Bs[:, j].mean()) >= abs(b0[j] - Bs[:, j].mean())).sum())/(B + 1))
        for j, nm in enumerate(["const", "same_city", "same_type", "same_macro", "log_dist"])]
print("\n[쌍 회귀: |y_i - y_j| ~ ...] (노드 순열 p)"); print(pd.DataFrame(rows, columns=["term", "coef", "perm_p"]).round(3).to_string(index=False))

# 같은 도시 내 분산이 무작위 배치보다 큰가
multi = [c for c in pd.unique(city) if (city == c).sum() > 1]
wss = lambda v: sum(((v[city == c] - v[city == c].mean())**2).sum() for c in multi)
ob = wss(y); sp = np.array([wss(RNG.permutation(y)) for _ in range(B)])
print(f"\n[같은 도시 내 SS] 관측 {ob:.3f} | 순열 평균 {sp.mean():.3f} | p(관측≥) = {(sp >= ob).mean():.3f}   (초안: p=.075)")

# =================== 2) 관문 근접도 민감도 ===================
GATEWAYS = {"Khorgos":(44.21,80.41),"Alashankou":(45.17,82.57),"Manzhouli":(49.58,117.38),"Erenhot":(43.65,111.98),"Dandong":(40.12,124.39),
 "Pingxiang":(22.11,106.76),"Mohan":(21.70,101.70),"Ningbo-Zhoushan":(29.93,121.88),"Shanghai-Yangshan":(30.63,122.07),"Shenzhen-Yantian":(22.55,114.25),
 "Guangzhou-Nansha":(22.73,113.60),"Qingdao":(36.07,120.32),"Tianjin port":(38.98,117.75),"Dalian port":(38.93,121.65),"Xiamen":(24.45,118.07)}
gw = pd.DataFrame(GATEWAYS, index=["lat", "lon"]).T.rename_axis("name").reset_index()
print("\n[관문 목록 - 좌표를 직접 검증하세요]"); print(gw.to_string(index=False))

dlt = PIV.loc[:, 2013:2019].mean(axis=1).values - PIV.loc[:, 2007:2012].mean(axis=1).values; kk = ~np.isnan(dlt)
def grad(g):
    gk = np.array([hav(la[i], lo[i], g.lat.values, g.lon.values).min() for i in range(n)])
    pr = stats.zscore(-np.log(gk)); sl = stats.linregress(pr[kk], dlt[kk]).slope
    X = pr[kk][np.argsort(RNG.random((B, kk.sum())), axis=1)]; Xc = X - X.mean(1, keepdims=True)
    s = (Xc @ (dlt[kk] - dlt[kk].mean()))/(Xc**2).sum(1)
    return sl, (1 + (np.abs(s) >= abs(sl)).sum())/(B + 1)
port = gw.name.str.contains(r"Ningbo|Yangshan|Yantian|Nansha|Qingdao|port|Xiamen", case=False)
var = {"all": gw, "land only": gw[~port], "ports only": gw[port], **{f"drop {m}": gw[gw.name != m] for m in gw.name}}
GW = pd.DataFrame([(a, *grad(v), len(v)) for a, v in var.items()], columns=["variant", "slope", "perm_p", "n_gateways"])
print("\n[관문 근접도 기울기: Δy(2013-19 vs 2007-12) ~ 근접도]"); print(GW.round(3).to_string(index=False))
print(f"\n요약: p 범위 [{GW.perm_p.min():.3f}, {GW.perm_p.max():.3f}] | 기울기 부호 {sorted(set(np.sign(GW.slope).astype(int)))} (초안: slope=-0.027, p=.35)")
GW.to_csv(OUT / "gateway_sensitivity.csv", index=False)

# ---- cell 10 ----
#

# ---- cell 11 ----
# =====================================================================
# Fig 5 (fig_R1_spatial_scale) & Fig 6 (map2_periods + core-periphery b,c) regenerated with n = 33
# Input : Transport_CN_Scholarly_Works.csv, gadm41_CHN_1.json (map only)
# Output: outputs_R1_v9_check/fig5_spatial_scale.(png|pdf), fig6_map_core_periphery.(png|pdf)
# Needs : pandas numpy scipy matplotlib geopandas
# =====================================================================
import io, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
warnings.filterwarnings("ignore")

BASE = Path("~/Research/TransDiff-CN_R1").expanduser()
OUT = BASE / "outputs_R1_v9_check"; OUT.mkdir(exist_ok=True)
GEOJSON = BASE / "gadm41_CHN_1.json"
SEED, EVENT, B = 20261122, 2013, 4999
RNG = np.random.default_rng(SEED)
FLOOR, SAME_CITY_KM = 25.0, 30.0
PERIODS = {"2007–12": (2007, 2012), "2013–19": (2013, 2019), "2020–23": (2020, 2023)}
norm = lambda s: re.sub(r"\s*\([^)]*\)\s*$", "", str(s)).strip()

META = {  # name: (city, lat, lon, region, type)
 "Chinese Academy of Sciences":("Beijing",39.9042,116.4074,"North","research"),"Tsinghua University":("Beijing",40.0033,116.3267,"North","C9"),
 "Beijing Jiaotong University":("Beijing",39.9647,116.3539,"North","transport"),"Tongji University":("Shanghai",31.2813,121.5034,"East","transport"),
 "Southeast University":("Nanjing",32.1578,118.8208,"East","C9"),"Zhejiang University":("Hangzhou",30.2587,120.1310,"East","C9"),
 "Shanghai Jiao Tong University":("Shanghai",31.2024,121.4337,"East","C9"),"Beihang University":("Beijing",40.0009,116.3428,"North","C7"),
 "Wuhan University of Technology":("Wuhan",30.5099,114.3974,"Central","transport"),"Central South University":("Changsha",28.1804,112.9321,"Central","transport"),
 "Harbin Institute of Technology":("Harbin",45.7761,126.6723,"Northeast","C9"),"University of Chinese Academy of Sciences":("Beijing",39.9975,116.3862,"North","research"),
 "Peking University":("Beijing",39.9971,116.3107,"North","C9"),"Jilin University":("Changchun",43.8878,125.3247,"Northeast","C9"),
 "Wuhan University":("Wuhan",30.5389,114.3599,"Central","C9"),"South China University of Technology":("Guangzhou",23.0490,113.4022,"South","C9"),
 "Shandong University":("Jinan",36.6804,117.0601,"North","C9"),"Sun Yat-sen University":("Guangzhou",23.0932,113.2978,"South","C9"),
 "Huazhong University of Science and Technology":("Wuhan",30.5091,114.4139,"Central","C9"),"Beijing Institute of Technology":("Beijing",39.9652,116.3223,"North","C7"),
 "Shenzhen University":("Shenzhen",22.5341,113.9298,"South","teaching"),"Southwest Jiaotong University":("Chengdu",30.6441,103.8418,"West","transport"),
 "Fudan University":("Shanghai",31.2985,121.5018,"East","C9"),"Sichuan University":("Chengdu",30.6310,104.0821,"West","C9"),
 "Tianjin University":("Tianjin",39.1053,117.1619,"North","C9"),"Dalian University of Technology":("Dalian",38.8574,121.5264,"Northeast","C9"),
 "Chongqing University":("Chongqing",29.5594,106.5509,"West","C9"),"Nanjing University":("Nanjing",32.1146,118.7835,"East","C9"),
 "Soochow University":("Suzhou",31.3019,120.6239,"East","teaching"),"Xi'an Jiaotong University":("Xi'an",34.2570,108.9784,"West","C9"),
 "University of Science and Technology of China":("Hefei",31.8315,117.2571,"East","C9"),"Shanghai University":("Shanghai",31.2835,121.4440,"East","teaching"),
 "Beijing University of Technology":("Beijing",39.8590,116.4815,"North","teaching")}
names = list(META); n = len(names)
city = np.array([META[k][0] for k in names]); lat = np.array([META[k][1] for k in names]); lon = np.array([META[k][2] for k in names])
is_bj = city == "Beijing"

# ---------------- panel (reconstructed from raw CSV; no zero filling) ----------------
raw = (BASE / "Transport_CN_Scholarly_Works.csv").read_bytes().decode("utf-8", errors="replace")
old = pd.read_csv(io.StringIO(raw), dtype=str, low_memory=False); old.columns = old.columns.str.strip(); del raw
old = old[old["Lens ID"].notna()].copy()
old["year"] = pd.to_numeric(old["Publication Year"], errors="coerce"); old["pat"] = pd.to_numeric(old["Citing Patents Count"], errors="coerce")
old = old[old.year.between(2007, 2023) & old.pat.notna()].copy(); old["year"] = old.year.astype(int)
old["inst"] = old.Institution.map(norm); old = old.drop_duplicates(["Lens ID", "inst"])
assert set(old.inst) == set(names), f"기관 불일치: {set(old.inst) ^ set(names)}"
PIV = np.log1p(old.groupby(["inst", "year"]).pat.mean()).unstack("year").reindex(names)      # y_it
NP = old.groupby(["inst", "year"]).size().unstack("year").reindex(names)                    # papers per cell
print("institutions:", n, "| non-empty cells:", int(PIV.notna().sum().sum()))

# ---------------- spatial helpers ----------------
def hav(la1, lo1, la2, lo2):
    p = np.pi / 180; a = np.sin((la2 - la1) * p / 2) ** 2 + np.cos(la1 * p) * np.cos(la2 * p) * np.sin((lo2 - lo1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))
D = hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
def rs(W): s = W.sum(1, keepdims=True); return np.divide(W, s, out=np.zeros_like(W), where=s > 0)
def W_floor(Dk): W = 1 / np.maximum(Dk, FLOOR); np.fill_diagonal(W, 0); return rs(W)
def moran(z, W):
    z = z - z.mean(); return len(z) / W.sum() * (z @ W @ z) / (z @ z)
def moran_perm(y, W, reps=B):
    z = y - y.mean(); I = len(z) / W.sum() * (z @ W @ z) / (z @ z)
    Z = z[np.argsort(RNG.random((reps, len(z))), axis=1)]
    sims = len(z) / W.sum() * ((Z @ W) * Z).sum(1) / (Z ** 2).sum(1)
    p = (1 + (np.abs(sims - sims.mean()) >= abs(I - sims.mean())).sum()) / (reps + 1)
    return I, p, np.percentile(sims, [2.5, 97.5])
ymean = lambda a, b: PIV.loc[:, a:b].mean(axis=1).values
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})

# =====================================================================
# FIG 5
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

# (a) Moran's I by year with 95% permutation band (25 km floor)
rows = []
for y in PIV.columns:
    v = PIV[y].values; k = ~np.isnan(v)
    if k.sum() < 12: continue
    I, p, (lo_, hi_) = moran_perm(v[k], W_floor(D[np.ix_(k, k)]), reps=999)
    rows.append((y, I, lo_, hi_, int(k.sum())))
MY = pd.DataFrame(rows, columns=["year", "I", "lo", "hi", "n"]).set_index("year")
ax[0].fill_between(MY.index, MY.lo, MY.hi, color="gray", alpha=.25, label="95% permutation band")
ax[0].plot(MY.index, MY.I, "o-", ms=3.5, label="Moran's I (25 km floor)")
ax[0].axhline(0, c="k", lw=.6); ax[0].axvline(EVENT, c="gray", ls=":")
ax[0].set_xlabel("Publication year"); ax[0].set_ylabel("Moran's I")
ax[0].set_title("(a) Moran's I by year", loc="left", fontweight="bold"); ax[0].legend(fontsize=7, frameon=False)
print("\n[5a] years outside band:", MY[(MY.I < MY.lo) | (MY.I > MY.hi)].index.tolist(), "| n per year:", MY.n.min(), "-", MY.n.max())

# (b) correlogram, 2013-19 means, distance bands
y1319 = ymean(*PERIODS["2013–19"]); k = ~np.isnan(y1319)
BANDS = [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 3000)]
rows = []
for lo_d, hi_d in BANDS:
    W = ((D > lo_d) & (D <= hi_d)).astype(float); np.fill_diagonal(W, 0); W = rs(W[np.ix_(k, k)])
    if W.sum() == 0: continue
    I, p, (a_, b_) = moran_perm(y1319[k], W)
    rows.append((f"{lo_d}–{hi_d}", I, a_, b_, p, int((W > 0).sum())))
CG = pd.DataFrame(rows, columns=["band_km", "I", "lo", "hi", "p", "n_links"])
xs = np.arange(len(CG)); ax[1].fill_between(xs, CG.lo, CG.hi, color="gray", alpha=.25, label="95% permutation band")
ax[1].plot(xs, CG.I, "o-", label="Moran's I (2013–19)"); ax[1].axhline(0, c="k", lw=.6)
ax[1].set_xticks(xs); ax[1].set_xticklabels(CG.band_km, fontsize=8); ax[1].set_xlabel("Distance band (km)")
ax[1].set_title("(b) Spatial correlogram, 2013–19", loc="left", fontweight="bold"); ax[1].legend(fontsize=7, frameon=False)
print("\n[5b] correlogram:\n", CG.round(3).to_string(index=False))

# (c) pairwise gap vs distance, 2013-19 means, same-city pairs highlighted
yk, Dk, ck = y1319[k], D[np.ix_(k, k)], city[k]; iu = np.triu_indices(len(yk), 1)
dist, gap = Dk[iu], np.abs(yk[:, None] - yk[None, :])[iu]; same = ck[iu[0]] == ck[iu[1]]
rho = stats.spearmanr(dist, gap)[0]
Pm = np.argsort(RNG.random((B, len(yk))), axis=1); G = np.abs(yk[Pm][:, iu[0]] - yk[Pm][:, iu[1]])
rg = stats.rankdata(G, axis=1); rg -= rg.mean(1, keepdims=True); rd = stats.rankdata(dist); rd -= rd.mean()
sims = (rg @ rd) / np.sqrt((rg ** 2).sum(1) * (rd ** 2).sum()); p_rho = (1 + (np.abs(sims) >= abs(rho)).sum()) / (B + 1)
ax[2].scatter(np.maximum(dist[~same], 1), gap[~same], s=10, c="tab:blue", alpha=.5, label="Different city")
ax[2].scatter(np.maximum(dist[same], 1), gap[same], s=22, c="tab:red", edgecolor="k", linewidth=.3, label="Same city")
ax[2].set_xscale("log"); ax[2].set_xlabel("Distance (km, log scale)"); ax[2].set_ylabel("|Δ institution mean|")
ax[2].set_title(f"(c) Outcome gap vs distance (ρ = {rho:.2f}, p = {p_rho:.2f}; n = {len(yk)})", loc="left", fontweight="bold", fontsize=9.5)
ax[2].legend(fontsize=7, frameon=False)
print(f"\n[5c] n={len(yk)} | Spearman rho = {rho:.3f} | perm p = {p_rho:.3f}  (본문: ρ=−0.03, p=.72)")
plt.tight_layout(); fig.savefig(OUT / "fig5_spatial_scale.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "fig5_spatial_scale.pdf", bbox_inches="tight"); plt.show()

# =====================================================================
# FIG 6: upper row = maps by period; lower row = Beijing vs others (dispersion, mean gap)
# =====================================================================
import geopandas as gpd
AEA = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs"
pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(AEA); px, py = pts.x.values, pts.y.values
prov = None
if GEOJSON.exists():
    prov = gpd.read_file(GEOJSON); prov = (prov.set_crs(4326) if prov.crs is None else prov).to_crs(AEA)
    prov["geometry"] = prov.geometry.simplify(2000); prov = prov.dissolve(by="NAME_1", as_index=False)
else: print("[map] GeoJSON 없음 → 경계 없이 점만 표시:", GEOJSON)

dev, npap, sdp = {}, {}, {}
for pn, (a, b) in PERIODS.items():
    v = ymean(a, b); dev[pn] = v - np.nanmean(v); sdp[pn] = np.nanstd(v, ddof=1)
    npap[pn] = NP.loc[:, a:b].sum(axis=1).values
vmax = max(np.nanmax(np.abs(d)) for d in dev.values()); nrm = TwoSlopeNorm(0, -vmax, vmax)
size = lambda m: 25 + 260 * np.sqrt(np.nan_to_num(m) / max(np.nanmax(list(npap.values())), 1))

fig = plt.figure(figsize=(17, 11)); gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1], hspace=.18, wspace=.12)
sc = None
for j, pn in enumerate(PERIODS):
    a = fig.add_subplot(gs[0, j])
    if prov is not None:
        prov.plot(ax=a, color="#f4f2ec", edgecolor="#bdb9ae", linewidth=.3, zorder=0)
        bb = prov.total_bounds; a.set_xlim(bb[0], bb[2]); a.set_ylim(bb[1], bb[3])
    a.set_aspect("equal"); a.set_axis_off()
    sc = a.scatter(px, py, s=size(npap[pn]), c=dev[pn], cmap="coolwarm", norm=nrm, edgecolor="k", linewidth=.4, alpha=.92, zorder=4)
    a.set_title(f"({'abc'[j]}) {pn}: SD of institution means = {sdp[pn]:.2f}", fontsize=10, loc="left", fontweight="bold")
    ia = a.inset_axes([0.02, 0.60, 0.30, 0.36])                      # Beijing inset
    if prov is not None: prov.plot(ax=ia, color="#f4f2ec", edgecolor="#bdb9ae", linewidth=.3, zorder=0)
    ia.scatter(px[is_bj], py[is_bj], s=size(npap[pn])[is_bj], c=dev[pn][is_bj], cmap="coolwarm", norm=nrm, edgecolor="k", linewidth=.4, zorder=4)
    ia.set_xlim(px[is_bj].min() - 4e4, px[is_bj].max() + 4e4); ia.set_ylim(py[is_bj].min() - 4e4, py[is_bj].max() + 4e4)
    ia.set_xticks([]); ia.set_yticks([]); ia.set_title("Beijing (8)", fontsize=7, pad=2)
cax = fig.add_axes([0.915, 0.60, 0.01, 0.22]); cb = fig.colorbar(sc, cax=cax); cb.set_label("Deviation from period mean (log scale)", fontsize=8)
print("\n[6a-c] SD of institution means by period:", {k: round(v, 3) for k, v in sdp.items()})

# lower row: Beijing group vs others
yrs = [y for y in PIV.columns if PIV.loc[is_bj, y].notna().sum() >= 3 and PIV.loc[~is_bj, y].notna().sum() >= 3]
sd_bj = [PIV.loc[is_bj, y].std(ddof=1) for y in yrs]; sd_ot = [PIV.loc[~is_bj, y].std(ddof=1) for y in yrs]
gap_m, gap_lo, gap_hi = [], [], []
for y in yrs:
    b_, o_ = PIV.loc[is_bj, y].dropna().values, PIV.loc[~is_bj, y].dropna().values
    gap_m.append(b_.mean() - o_.mean())
    bs = [RNG.choice(b_, len(b_)).mean() - RNG.choice(o_, len(o_)).mean() for _ in range(1000)]
    gap_lo.append(np.percentile(bs, 2.5)); gap_hi.append(np.percentile(bs, 97.5))
a4 = fig.add_subplot(gs[1, 0]); a5 = fig.add_subplot(gs[1, 1])
a4.plot(yrs, sd_bj, "o-", ms=3.5, c="tab:red", label="Beijing group (n = 8)"); a4.plot(yrs, sd_ot, "s-", ms=3.5, c="tab:blue", label="Other institutions (n = 25)")
a4.axvline(EVENT, c="gray", ls=":"); a4.set_xlabel("Publication year"); a4.set_ylabel("SD of institution means")
a4.set_title("(d) Dispersion: Beijing group vs others", loc="left", fontweight="bold"); a4.legend(fontsize=7, frameon=False)
a5.fill_between(yrs, gap_lo, gap_hi, color="gray", alpha=.25, label="95% bootstrap interval (institutions)")
a5.plot(yrs, gap_m, "o-", ms=3.5, c="k", label="Mean gap: Beijing minus others"); a5.axhline(0, c="k", lw=.6); a5.axvline(EVENT, c="gray", ls=":")
a5.set_xlabel("Publication year"); a5.set_ylabel("Mean gap (log scale)"); a5.set_title("(e) Core–non-core mean gap", loc="left", fontweight="bold"); a5.legend(fontsize=7, frameon=False)
post = [i for i, y in enumerate(yrs) if y >= EVENT]
print(f"[6e] mean gap after {EVENT}: {np.mean([gap_m[i] for i in post]):.3f} | years with CI excluding 0: {[yrs[i] for i in range(len(yrs)) if gap_lo[i] > 0 or gap_hi[i] < 0]}")
fig.savefig(OUT / "fig6_map_core_periphery.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "fig6_map_core_periphery.pdf", bbox_inches="tight"); plt.show()
print("\n저장:", OUT)

# ---- cell 12 ----
#

# ---- cell 13 ----

