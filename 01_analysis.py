# ------------------------------------------------------------
# Cyber Crime Analysis
# Author: gayoung
# Date: 2025-11-11
# Description:
#   경찰청 사이버범죄 통계 + KOSIS 인터넷 이용률 + 신뢰지수 데이터 병합 및 분석
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import re

# ------------------------------------------------------------
# 1. 공통 유틸 함수
# ------------------------------------------------------------
def load_csv_safely(filename: str) -> pd.DataFrame:
    """CSV 파일을 자동으로 인코딩 감지하여 안전하게 불러오기"""
    encodings = ["utf-8-sig", "cp949", "utf-8"]
    for enc in encodings:
        try:
            df = pd.read_csv(filename, encoding=enc)
            print(f" {filename} 불러오기 성공 (encoding={enc})")
            return df
        except Exception:
            continue
    raise ValueError(f"  {filename} 파일을 읽을 수 없습니다. (인코딩 문제)")

# ------------------------------------------------------------
# 2. 경찰청 사이버범죄 통계
# ------------------------------------------------------------
print(" 경찰청 사이버범죄 통계 불러오는 중...")
crime = load_csv_safely("police_cybercrime.csv")

# 데이터 정제
crime.columns = crime.columns.str.strip()
crime = crime.rename(columns={crime.columns[0]: "year", crime.columns[1]: "total_crime"})
crime["year"] = pd.to_numeric(crime["year"], errors="coerce")
crime = crime.dropna(subset=["year"])
crime["year"] = crime["year"].astype(int)

print("\n 경찰청 사이버범죄 미리보기:")
print(crime.head())
# ------------------------------------------------------------
#  3. KOSIS 인터넷 이용률 데이터
# ------------------------------------------------------------
print("\n KOSIS 인터넷 이용률 데이터 불러오는 중...")
internet = load_csv_safely("internet_use.csv")

print("\n  인터넷 이용률 파일의 열 이름:")
print(list(internet.columns))

# "전체" 행만 선택
first_col = internet.columns[0]
internet = internet[internet[first_col].astype(str).str.contains("전체", na=False)]

# 숫자형 컬럼 중 짝수번째 (이용률만)
numeric_cols = [col for col in internet.columns if re.match(r"^\d{4}(\.\d+)?$", str(col))]
even_cols = [col for i, col in enumerate(numeric_cols) if i % 2 == 0]

# wide → long 변환
internet_long = pd.melt(
    internet,
    id_vars=[first_col],
    value_vars=even_cols,
    var_name="year",
    value_name="internet_rate"
)

internet_long["year"] = internet_long["year"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
internet_long["internet_rate"] = pd.to_numeric(internet_long["internet_rate"], errors="coerce")
internet = internet_long[["year", "internet_rate"]].dropna()

print("\n 인터넷 이용률 미리보기:")
print(internet.head())

# ------------------------------------------------------------
#  4. 신뢰지수 데이터 (KOSIS)
# ------------------------------------------------------------
print("\n 신뢰지수 데이터 불러오는 중...")

trust_raw = pd.read_csv("trust_index.csv", encoding="utf-8-sig", header=None)

# 연도는 3번째 행(인덱스 2)
year_row = trust_raw.iloc[2, 2:].dropna().astype(str).tolist()

# "전체" 행(인덱스 3)의 값들만 가져오기
trust_total = trust_raw.iloc[3, 2:].astype(float).tolist()

# DataFrame 생성
trust = pd.DataFrame({
    "year": [int(y) for y in year_row],
    "trust_index": trust_total
})

print("\n 신뢰지수 정리 완료 미리보기:")
print(trust.head(10))


# ------------------------------------------------------------
# 5. 데이터 병합
# ------------------------------------------------------------
print("\n  데이터 병합 중...")

crime["year"] = pd.to_numeric(crime["year"], errors="coerce").astype("Int64")
internet["year"] = pd.to_numeric(internet["year"], errors="coerce").astype("Int64")
trust["year"] = pd.to_numeric(trust["year"], errors="coerce").astype("Int64")

print("\n 데이터 타입 확인:")
print("crime:", crime.dtypes)
print("internet:", internet.dtypes)
print("trust:", trust.dtypes)

merged = crime.merge(internet, on="year", how="left").merge(trust, on="year", how="left")
merged = merged[(merged["year"] >= 2015) & (merged["year"] <= 2023)].sort_values("year").reset_index(drop=True)

print("\n 병합된 데이터 미리보기:")
print(merged)

# ------------------------------------------------------------
#  6. 결측값 확인
# ------------------------------------------------------------
print("\n  결측값 통계:")
print(merged.isna().sum())

# ------------------------------------------------------------
#    시각화
# ------------------------------------------------------------
plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 폰트
plt.figure(figsize=(10, 6))
sns.lineplot(data=merged, x="year", y="total_crime", label="사이버범죄 건수")
sns.lineplot(data=merged, x="year", y="internet_rate", label="인터넷 이용률(%)")
sns.lineplot(data=merged, x="year", y="trust_index", label="신뢰지수")
plt.title("사이버범죄, 인터넷 이용률, 신뢰지수 추이 (2015~2023)")
plt.xlabel("연도")
plt.ylabel("값")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
#     상관관계 분석
# ------------------------------------------------------------
print("\n  상관관계 분석:")
print(merged.corr(numeric_only=True))

# ------------------------------------------------------------
# 5. 회귀분석
# ------------------------------------------------------------
print("\n  회귀분석 수행 중...")

import statsmodels.api as sm

# 분석용 데이터 필터링: '발생건수'만 사용
df_analysis = merged[merged["total_crime"] == "발생건수"].copy()

# 분석 변수
df_analysis["total_crime"] = df_analysis["사이버사기_직거래"] + df_analysis["사이버사기_쇼핑몰"] + df_analysis["사이버사기_게임"]

# 숫자형 변환 (NaN 자동 처리)
df_analysis["internet_rate"] = pd.to_numeric(df_analysis["internet_rate"], errors="coerce")
df_analysis["trust_index"] = pd.to_numeric(df_analysis["trust_index"], errors="coerce")

# 결측 제거
df_analysis = df_analysis.dropna(subset=["internet_rate", "trust_index", "total_crime"])

# 회귀모형 구성
X = df_analysis[["internet_rate", "trust_index"]]
X = sm.add_constant(X)  # 상수항 추가
y = df_analysis["total_crime"].astype(float)

# 회귀분석 수행
model = sm.OLS(y, X).fit()

print("\n 회귀분석 결과:")
print(model.summary())
