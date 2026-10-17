# 분석 코드 안내

최신 연구목적, 자료 기준, OLS 최종모형과 A3 Ridge Logistic 결과는 상위 [`README.md`](../README.md)를 참고한다.

## 최신 분석 순서

1. `analysis/011_새데이터_승차변화_시각화.ipynb`
2. `analysis/11-8_평균통행시간_포함_기본모형_OLS.ipynb`
3. `analysis/12-1_GTX-A_수요비중증가_Ridge_Logistic.ipynb`
4. `analysis/12-2_Ridge_Logistic_예측성능_개선실험.py`
5. `analysis/12-3_기초수요변수_제한비교.py`
6. `analysis/12-4_변수선택포함_중첩교차검증.py`

## 현재 상태

- OLS 최종모형: 126개 격자의 승차인원 변화량을 종속변수로 사용하고 보행 접근성, 인구, 경쟁 목적지 비율, 버스 통행시간, IC 거리, 주거지역 비율, 건축물 연면적을 투입한다.
- 머신러닝 우선 후보: A3 Ridge Logistic Regression
- A3 입력변수: 개통 전 역 권역 내 수요 비중, `log(1+개통 전 승차인원)`, GTX역 보행시간, GTX 경쟁 목적지 비율
- 중첩 Leave-One-Station-Out 전체 OOF ROC-AUC: 0.6548
- GTX-B·C 적용 결과: 확정적인 0·1 판정보다 비중 확대 가능성 점수와 상대적 순위로 제시할 계획이다.

이전 셀×노선 Poisson QMLE·NB2 코드는 과거 분석 재현을 위해 유지한다.
