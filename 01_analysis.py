import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib import font_manager, rc
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------------------------------------
# 1. 환경 설정 및 한글 폰트 (Windows 기준)
# ------------------------------------------------------------
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)    
except:
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------
# 2. 데이터 로드 및 컬럼명 영문 표준화 (Consistency)
# ------------------------------------------------------------
# 경찰청 데이터 컬럼 매핑 (가독성 및 유지보수용 영문명)
col_mapping = {
    "year": "year", "gubun": "gubun",
    "hacking_account": "해킹_계정도용", "hacking_intrusion": "해킹_단순침입",
    "malware_ransom": "악성프로그램_랜섬웨어", "malware_etc": "악성프로그램_기타",
    "fraud_direct": "사이버사기_직거래", "fraud_shop": "사이버사기_쇼핑몰",
    "fraud_game": "사이버사기_게임", "fraud_etc": "사이버사기_기타",
    "finance_phishing": "사이버금융_피싱", "finance_smishing": "사이버금융_스미싱",
    "finance_messenger": "사이버금융_메신저사기",
    "defamation": "사이버 명예훼손_모욕"
}

# CSV 로드 (헤더 깨짐 방지를 위해 직접 리스트 지정)
# 실제 파일의 모든 컬럼을 순서대로 적어야 하므로, 중요 컬럼 위주로 슬라이싱하거나 전체 리스트를 사용합니다.
all_cols = [
    "year", "gubun", "hacking_account", "hacking_intrusion", "hacking_data_leak", "hacking_data_damage",
    "dos", "malware_ransom", "malware_etc", "net_infringement_etc",
    "fraud_direct", "fraud_shop", "fraud_game", "fraud_email", "fraud_etc",
    "finance_phishing", "finance_farming", "finance_smishing", "finance_memory", "finance_messenger", "finance_finance_etc",
    "location_infringement", "copyright_infringement", "net_use_etc",
    "porn_general", "porn_child", "porn_illegal_video",
    "gambling_sports", "gambling_horse", "gambling_casino", "gambling_etc",
    "defamation", "stalking", "illegal_content_etc"
]

crime_df = pd.read_csv("police_cybercrime.csv", encoding="cp949", skiprows=1, names=all_cols)
crime_gen = crime_df[crime_df["gubun"] == "발생건수"].copy()

# ------------------------------------------------------------
# 3. 데이터 공백 메우기 (인터넷 이용률 2015-2023)
# ------------------------------------------------------------
# internet_use.csv에 과거 데이터가 없을 경우를 대비한 KOSIS 공식 수치(전체 이용률)
official_internet_rates = {
    2015: 85.1, 2016: 88.3, 2017: 90.3, 2018: 91.5, 
    2019: 91.8, 2020: 91.9, 2021: 93.0, 2022: 93.0, 2023: 94.0
}
internet = pd.DataFrame(list(official_internet_rates.items()), columns=['year', 'internet_rate'])

# 신뢰지수 (KOSIS 대인신뢰도: 0~100점 척도 환산)
trust_raw = pd.read_csv("trust_index.csv", encoding="utf-8-sig", header=None)
year_row = [int(y) for y in trust_raw.iloc[2, 2:].dropna().values]
trust_val = [float(v) for v in trust_raw.iloc[3, 2:].dropna().values]
trust = pd.DataFrame({"year": year_row, "trust_index": trust_val})

# ------------------------------------------------------------
# 4. 데이터 통합 및 변수 생성 (조절효과 & 시차)
# ------------------------------------------------------------
merged = crime_gen.merge(internet, on="year", how="inner").merge(trust, on="year", how="inner")
merged = merged.sort_values("year")

# 종속변수 생성 (연구 주제별 범죄 그룹화)
merged['crime_fraud'] = merged[['fraud_direct', 'fraud_shop', 'fraud_game', 'fraud_etc']].sum(axis=1)
merged['crime_infringe'] = merged[['hacking_account', 'hacking_intrusion', 'malware_ransom']].sum(axis=1)
merged['crime_defame'] = merged['defamation']

# 시차 변수 (전년도 신뢰가 올해 범죄에 미치는 영향)
merged['trust_lag1'] = merged['trust_index'].shift(1)

# 조절효과 변수 (Centering 후 상호작용항 생성)
merged['trust_c'] = merged['trust_index'] - merged['trust_index'].mean()
merged['internet_c'] = merged['internet_rate'] - merged['internet_rate'].mean()
merged['interaction'] = merged['trust_c'] * merged['internet_c']

merged_clean = merged.dropna().reset_index(drop=True)

# ------------------------------------------------------------
# 3. 모델 정밀 진단 함수 (VIF 포함)
# ------------------------------------------------------------
def diagnose_model(target_crime, df):
    print(f"\n" + "="*50)
    print(f" [{target_crime}] 모델 정밀 진단")
    print("="*50)
    
    # 상호작용항을 포함한 독립변수 설정
    X = df[['trust_c', 'internet_c', 'interaction']]
    X = sm.add_constant(X)
    y = df[target_crime]
    
    model = sm.OLS(y, X).fit()
    
    # VIF 계산
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print(model.summary())
    print("\n[VIF 지수 - 10 이상이면 다중공선성 위험]")
    print(vif)
    return model

# # ------------------------------------------------------------
# # 5. 가설 검증용 회귀 분석 수행 함수
# # ------------------------------------------------------------
# def run_analysis(target_crime, df):
#     print(f"\n" + "="*50)
#     print(f" 분석 대상 범죄: {target_crime}")
#     print("="*50)
    
#     # 가설: 신뢰지수(IV) + 인터넷이용률(Moderator) + 상호작용항
#     X = df[['trust_index', 'internet_rate', 'interaction']]
#     X = sm.add_constant(X)
#     y = df[target_crime]
    
#     model = sm.OLS(y, X).fit()
#     print(model.summary())
#     return model

# 모든 범죄 유형 진단 실행
diag_fraud = diagnose_model('crime_fraud', merged_clean)
diag_infringe = diagnose_model('crime_infringe', merged_clean)
diag_defame = diagnose_model('crime_defame', merged_clean)
# ------------------------------------------------------------
# 6. 시각적 추세 분석 (이중 축 그래프)
# ------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(merged['year'], merged['trust_index'], color='blue', marker='o', label='사회적 신뢰(대인신뢰도)')
ax1.set_ylabel('신뢰 지수', color='blue')

ax2 = ax1.twinx()
ax2.plot(merged['year'], merged['crime_fraud'], color='red', marker='s', linestyle='--', label='사이버 사기 건수')
ax2.set_ylabel('사이버 범죄 발생 건수', color='red')

plt.title("사회적 신뢰와 사이버 사기 범죄의 시계열적 상관관계 (2015-2023)")
ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()

# ------------------------------------------------------------
# 6. 시각적 추세 분석 (이중 축 그래프)
# ------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(merged['year'], merged['trust_index'], color='blue', marker='o', label='사회적 신뢰(대인신뢰도)')
ax1.set_ylabel('신뢰 지수', color='blue')

ax2 = ax1.twinx()
ax2.plot(merged['year'], merged['crime_defame'], color='red', marker='s', linestyle='--', label='사이버 사기 건수')
ax2.set_ylabel('사이버 범죄 발생 건수', color='red')

plt.title("사회적 신뢰와 비방형(명예훼손/비방) 범죄의 시계열적 상관관계 (2015-2023)")
ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()

# ------------------------------------------------------------
# 6. 시각적 추세 분석 (이중 축 그래프)
# ------------------------------------------------------------
# 시각화 (신뢰지수 vs 침해형 범죄)
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(merged['year'], merged['trust_index'], color='blue', marker='o', label='사회적 신뢰')
ax1.set_ylabel('신뢰 지수', color='blue')

ax2 = ax1.twinx()
ax2.plot(merged['year'], merged['crime_infringe'], color='green', marker='^', linestyle=':', label='침해형 범죄(해킹 등)')
ax2.set_ylabel('침해형 범죄 발생 건수', color='green')

plt.title("사회적 신뢰와 침해형(해킹/악성코드) 범죄의 관계 (2015-2023)")
ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()

# ------------------------------------------------------------
# 7. 기술통계량 및 상관관계 분석 (추가)
# ------------------------------------------------------------

# 1) 기술통계량 출력
desc_stats = merged_clean[['trust_index', 'internet_rate', 'crime_fraud', 'crime_infringe', 'crime_defame']].describe()
print("\n" + "="*50)
print(" [기술통계량 분석 결과] ")
print("="*50)
print(desc_stats.transpose()[['mean', 'std', 'min', 'max']])

# 2) 상관관계 분석
correlation_matrix = merged_clean[['trust_index', 'internet_rate', 'crime_fraud', 'crime_infringe', 'crime_defame']].corr()
print("\n" + "="*50)
print(" [상관관계 분석 결과 (Pearson)] ")
print("="*50)
print(correlation_matrix)

# 3) 상관관계 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("주요 변수 간 상관관계 히트맵")
plt.show()

# print(f" [{target_crime}] 모델 정밀 진단")
print("="*50)
    