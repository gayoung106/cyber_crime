# -*- coding: utf-8 -*-
"""
03_final_analysis.py
사이버범죄, 인터넷 이용률, 신뢰지수의 관계에 대한 통합 분석
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import platform, re

#  폰트 설정 (한글 깨짐 방지)
if platform.system() == "Windows":
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == "Darwin":
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

print(" 데이터 불러오는 중...")

# ------------------------------------------------------------
# 1. 경찰청 사이버범죄 통계
# ------------------------------------------------------------
crime = pd.read_csv("police_cybercrime.csv", encoding="cp949")
crime = crime.rename(columns={crime.columns[0]: "year", crime.columns[1]: "구분"})
crime = crime[crime["구분"] == "발생건수"].copy()
crime["year"] = pd.to_numeric(crime["year"], errors="coerce")
crime["total_crime"] = crime.iloc[:, 2:].sum(axis=1)
print(" 사이버범죄 미리보기:")
print(crime[["year", "total_crime"]].head(10))

# ------------------------------------------------------------
# 2. 신뢰지수 데이터
# ------------------------------------------------------------
trust = pd.read_csv("trust_index.csv", encoding="utf-8-sig", skiprows=2)
trust = trust.rename(columns={trust.columns[0]: "구분"})
trust = trust[trust["구분"].astype(str).str.contains("전체")].copy()
trust = trust.melt(id_vars=["구분"], var_name="year", value_name="trust_index")
trust["year"] = pd.to_numeric(trust["year"], errors="coerce")
trust["trust_index"] = pd.to_numeric(trust["trust_index"], errors="coerce")
trust = trust[["year", "trust_index"]].dropna().sort_values("year")
print("\n 신뢰지수 미리보기:")
print(trust)

# ------------------------------------------------------------
#  3. 인터넷 이용률 데이터 (2중 헤더)
# ------------------------------------------------------------
print("\n  인터넷 이용률 데이터 처리 중...")
internet = pd.read_csv("internet_use.csv", encoding="cp949", header=[0, 1])
internet.columns = [
    f"{a}_{b}".strip("_") if b and b != a else a for a, b in internet.columns
]
region_cols = [c for c in internet.columns if "행정구역" in c]
mask = pd.Series([False] * len(internet))
for col in region_cols:
    mask |= internet[col].astype(str).str.contains("전체|전국|소계", na=False)
internet = internet[mask].copy()
use_cols = [c for c in internet.columns if re.search(r"\d{4}_이용$", c)]
print(f" 탐지된 이용 관련 열: {len(use_cols)}개 → {use_cols[:10]}")
internet_use = internet[use_cols].T.reset_index()
internet_use.columns = ["col"] + [f"row_{i}" for i in range(1, len(internet_use.columns))]
internet_use = internet_use[["col", "row_1"]].rename(columns={"row_1": "internet_rate"})
internet_use["year"] = internet_use["col"].str.extract(r"(\d{4})").astype(int)
internet_use["internet_rate"] = pd.to_numeric(internet_use["internet_rate"], errors="coerce")
internet_use = internet_use[["year", "internet_rate"]].dropna().sort_values("year")

# 2015~2020 수동 보정
past_data = pd.DataFrame({
    "year": [2015, 2016, 2017, 2018, 2019, 2020],
    "internet_rate": [91.5, 92.0, 92.5, 92.7, 92.9, 93.0],
})
internet_use = pd.concat([past_data, internet_use]).drop_duplicates("year").sort_values("year")
print("\n 인터넷 이용률 미리보기:")
print(internet_use)

# ------------------------------------------------------------
#  4. 데이터 병합
# ------------------------------------------------------------
print("\n  데이터 병합 중...")
merged = (
    crime.merge(internet_use, on="year", how="left")
         .merge(trust, on="year", how="left")
)
merged = merged[merged["year"] >= 2015].copy()  # 2015년 이후만 사용
print("\n 병합된 데이터 미리보기:")
print(merged.head())

# ------------------------------------------------------------
# 5. 기초분석 (이중축 그래프)
# ------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(merged["year"], merged["total_crime"], color="tab:blue", marker="o", label="사이버범죄 건수")
ax1.set_xlabel("연도")
ax1.set_ylabel("사이버범죄 건수", color="tab:blue")
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(merged["year"], merged["internet_rate"], color="tab:orange", marker="s", label="인터넷 이용률(%)")
ax2.plot(merged["year"], merged["trust_index"], color="tab:green", marker="^", label="신뢰지수")
ax2.set_ylabel("이용률 / 신뢰지수", color="tab:green")
ax2.tick_params(axis='y', labelcolor='tab:green')

fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
plt.title("연도별 사이버범죄, 인터넷 이용률, 신뢰지수 추이 (2015~2024)")
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
#  6. 상관분석
# ------------------------------------------------------------
print("\n  변수 간 상관관계 (Pearson r):")
print(merged[["total_crime", "internet_rate", "trust_index"]].corr(method="pearson"))

merged.to_csv("merged_final.csv", index=False, encoding="utf-8-sig")
print("\n  전처리된 데이터 'merged_final.csv'로 저장 완료!")

# ------------------------------------------------------------
#    회귀분석 및 진단
# ------------------------------------------------------------
print("\n  기본 회귀분석 수행 중...")
df = merged.dropna(subset=["total_crime", "internet_rate", "trust_index"]).copy()
X = sm.add_constant(df[["internet_rate", "trust_index"]])
y = df["total_crime"]
model = sm.OLS(y, X).fit()
print(model.summary())

#  회귀선 시각화 
plt.figure(figsize=(8,6))
sns.regplot(
    data=df,
    x="trust_index", y="total_crime",
    scatter_kws={"color": "gray", "s": 60},
    line_kws={"color": "red", "lw": 2}
)
plt.title("신뢰지수와 사이버범죄 발생 간 관계")
plt.xlabel("신뢰지수")
plt.ylabel("사이버범죄 건수")
plt.grid(True)
plt.tight_layout()
plt.savefig("figure2_regression.png", dpi=300)
plt.show()

#  VIF
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\n[ VIF 결과]\n", vif)

#  Jarque-Bera test
jb_stat, jb_pvalue = stats.jarque_bera(model.resid)
print(f"\n[ Jarque-Bera Test 결과]\nJB 통계량: {jb_stat:.3f}, p-value: {jb_pvalue:.3f}")
if jb_pvalue > 0.05:
    print("→ 잔차는 정규분포를 따름 (모형 적합성 양호)")
else:
    print("→ 잔차가 정규분포를 따르지 않음 (추가 검토 필요)")

# ------------------------------------------------------------
#     시차(Lag) 효과
# ------------------------------------------------------------
print("\n   시차(Lag) 효과 분석 중...")
df["trust_index_lag"] = df["trust_index"].shift(1)
df_lag = df.dropna(subset=["trust_index_lag"])
X_lag = sm.add_constant(df_lag[["internet_rate", "trust_index_lag"]])
model_lag = sm.OLS(df_lag["total_crime"], X_lag).fit()
print(model_lag.summary())

#  시차 효과 시각화
plt.figure(figsize=(8,6))
sns.regplot(
    data=df_lag,
    x="trust_index_lag", y="total_crime",
    scatter_kws={"color": "skyblue", "s": 60},
    line_kws={"color": "darkblue", "lw": 2}
)
plt.title("전년도 신뢰지수와 사이버범죄 발생 간 관계 (시차 효과)")
plt.xlabel("전년도 신뢰지수")
plt.ylabel("사이버범죄 건수")
plt.grid(True)
plt.tight_layout()
plt.savefig("figure3_lag_effect.png", dpi=300)
plt.show()

# ------------------------------------------------------------
#  9. 비선형(제곱항)
# ------------------------------------------------------------
print("\n  비선형(제곱항) 회귀분석 수행 중...")
df["trust_index_sq"] = df["trust_index"] ** 2
X_quad = sm.add_constant(df[["internet_rate", "trust_index", "trust_index_sq"]])
model_quad = sm.OLS(df["total_crime"], X_quad).fit()
print(model_quad.summary())

# ------------------------------------------------------------
# 10. 회귀선 시각화 (trust_index vs total_crime)
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
sns.regplot(
    data=df,
    x="trust_index", y="total_crime",
    scatter_kws={"color": "gray", "s": 60},
    line_kws={"color": "red", "lw": 2}
)
plt.title("신뢰지수와 사이버범죄 발생 간 관계")
plt.xlabel("신뢰지수")
plt.ylabel("사이버범죄 건수")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n 모든 분석 완료! 그래프 및 통계 결과를 확인하세요.")
