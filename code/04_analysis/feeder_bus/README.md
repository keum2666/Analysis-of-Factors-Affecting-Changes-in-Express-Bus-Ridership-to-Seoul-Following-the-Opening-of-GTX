# GTX-A 역행 마을·시내버스 분석

이 폴더는 2026-10-20 전달받은 `마을버스_노트북이전`의 현재 s10 분석 코드 11개다. 전달본에는 ipynb는 없으며 Python 스크립트가 들어 있다. 기존 광역버스 7변수 OLS는 수정하지 않았다.

## 위치

- 현재 코드: `code/04_analysis/feeder_bus/`
- 마을·시내버스 수집 원본: `data/raw/gtx_a/feeder_bus/transport_card/`
- 정류장 매칭 자료: `data/processed/gtx_a/feeder_bus/stops/`
- 공통 126개 기초표: `data/final/gtx_a/S1_126격자_건수기준_기초표.csv`
- 철도역 수 등 보조 입력: `data/processed/gtx_a/feeder_bus/support/`
- 정류장·노선 parquet: `data/raw/common/smartcard_20241017/`
- 결과표: `results/analysis/feeder_bus/tables/`
- 그림: `results/analysis/feeder_bus/figures/`
- 수집 로그: `results/analysis/feeder_bus/logs/` (`raw/_수집기록.csv`는 이 파일의 상대 심볼릭 링크)
- 설명 문서: `docs/methodology/feeder_bus/`
- 과거 광역버스 예측 실험 등 전달 부속자료: `archive/research_history/feeder_bus_transfer_20261020/`
- 경로 수정 전 s10 원본: `archive/transfer_code_originals/feeder_bus_20261020/`
- 통합 및 검증 내역: `docs/reports/feeder_bus_import_20261020/`

## 실행

UROP 최상위에서 현재 smartcity Python을 사용할 경우:

```zsh
/Users/keumkyounghoon/Desktop/tool/envs/smartcity/bin/python code/04_analysis/feeder_bus/s10_12_feeder_final.py
```

기존 자료를 이용한 주모형과 점검표를 다시 생성한다. 직접 실행하면 현재 결과표를 덮어쓰므로 결과 보존이 필요하면 먼저 복사하거나 `UROP_FEEDER_TEST_RESULTS`를 임시 결과 디렉터리로 설정한다. 경로는 스크립트 위치 기준으로 찾으므로 작업 디렉터리가 달라도 된다. 원래 전달 폴더와 Downloads는 필요하지 않다.

전체 전처리 순서:

1. `s10_6_feeder_analysis.py`
2. `s10_8_station_name_ols.py`
3. `s10_10_peak_matching.py`
4. `s10_11_feeder_dv_same_supply.py`
5. `s10_14_pop2025.py`
6. `s10_12_feeder_final.py`
7. `s10_13_feeder_eda_vars.py`
8. `s10_15_spearman_fig.py`

STTN만 요구하는 스크립트는 인자가 없으면 포함된 `data/raw/common/smartcard_20241017/STTN_20241017.parquet`를 쓴다. 필요하면 첫 인자로 다른 STTN 경로를 명시할 수 있다. `s10_1`, `s10_3`은 미포함 전국 TCN 원본을 요구하므로 위 실행 순서에서 제외했다. `s10_4_collect_feeder.py`는 API 수집용이므로 통합 과정에서 실행하지 않았다. API 키는 UROP 최상위 `.env` 또는 환경변수로만 읽는다.

## 검증 범위와 제한

- 현재 스크립트 11개와 경로 도우미의 문법 검사 통과.
- 원본 AST와 비교하여 계산 로직 유지 확인. 변경은 경로, STTN 기본 경로, 한글 글꼴 선택뿐이다.
- 코드에 명시된 경로 57개 모두 존재 확인.
- 임시 작업공간에서 `s10_14`, `s10_12`, `s10_15` 실제 실행 성공. 주 OLS·인구 결과 재현 확인.
- `s10_6`, `s10_8`, `s10_10`, `s10_11`, `s10_13`은 현재 smartcity 환경에 pyarrow가 없어 import 단계에서 중단했다. 전체 파이프라인 실행 성공으로 보지 않는다.
- pyarrow 다운로드는 현재 실행 환경의 pypi.org DNS 접속 실패로 완료하지 못했다. 인터넷이 되는 터미널에서 아래 명령으로 설치 후 위 전처리 실행을 확인해야 한다.

```zsh
/Users/keumkyounghoon/Desktop/tool/envs/smartcity/bin/python -m pip install pyarrow
```

원본의 날짜·126개 표본·종속변수·표준오차·보조분석 제외 기준 등 연구 설정은 바꾸지 않았다. 문서에 기재된 92개 공급구조 유지 점검은 운정 전부를 포함하는 원본 조건이며 광역버스 126개 선정과 구별해야 한다.
