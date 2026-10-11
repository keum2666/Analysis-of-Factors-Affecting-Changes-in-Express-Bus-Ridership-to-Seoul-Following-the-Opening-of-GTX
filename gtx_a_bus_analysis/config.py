"""경로와 공통 함수. processing, analysis의 모든 노트북이 이 파일을 불러온다.

데이터는 모두 이 폴더 안의 data/ 에 있으므로 폴더를 통째로 옮겨도 그대로 동작한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# 경로 (폴더 구조는 README.md 참고)
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parent  # 이 폴더를 통째로 옮겨도 경로가 유지된다
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# data/raw — 원자료 (읽기 전용)
RAW_PATHS = {
    2024: DATA_DIR / "raw" / "transport_card" / "gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv",
    2025: DATA_DIR / "raw" / "transport_card" / "gtx_a_transport_card_20251016_raw_with_coords.csv",
}
ADMIN_PATH = DATA_DIR / "raw" / "boundaries" / "HangJeongDong_ver20260701.geojson"
GRID_PATH = DATA_DIR / "raw" / "grid_250m" / "빈격자(250m).shp"
SGIS_DIR_2024 = DATA_DIR / "raw" / "sgis_2024"  # 고양인구 / 파주인구 / 건축물수
SGIS_DIR_2025 = DATA_DIR / "raw" / "sgis_2025"  # 고양인구 / 파주인구
SGIS_DIR = SGIS_DIR_2024  # 기존 03_독립변수_인구_건축물 노트북과의 호환용

# data/processed — 모든 중간·최종 분석용 집계자료
STABLE_PATH = PROCESSED_DIR / "01_공급구조유지_250m격자.csv"
WALK_PATH = PROCESSED_DIR / "02_GTX역_보행접근성_250m격자.csv"
IC_PATH = PROCESSED_DIR / "03_IC_도로접근성_250m격자.csv"

# _원본전달본 — 받은 원본 결과 (정리본과 대조용)
ORIGINAL_OUTPUT = PROJECT_DIR / "_원본전달본" / "Analisis3" / "output"

# processing 결과는 모두 data/processed에 저장
OUT1 = PROCESSED_DIR
DEPENDENT_PATH = OUT1 / "04_광역버스_승차인원변화_250m격자.csv"
# 최종 종일 모형용 셀×노선 GTX 경쟁 목적지 비율
COMPETING_CELL_ROUTE_PATH = OUT1 / "05_GTX_경쟁목적지비율_종일_셀노선.csv"
COMPETING_CELL_PATH = OUT1 / "06_GTX_경쟁목적지비율_종일_셀.csv"
POPULATION_PATH = OUT1 / "07_인구_2024_250m격자.csv"
BUILDING_PATH = OUT1 / "08_건축물수_2024_250m격자.csv"
MODEL_DATA_PATH = OUT1 / "09_회귀분석_격자통합데이터.csv"
CELL_ROUTE_ALL_DAY_PATH = PROCESSED_DIR / "10_셀노선별_종일승차인원.csv"
POPULATION_CHANGE_PATH = OUT1 / "11_20~50대인구변화_2024_2025_250m격자.csv"
LAND_PRICE_2024_PATH = OUT1 / "12_공시지가_2024_250m격자.csv"
TRAVEL_TIME_2024_PATH = OUT1 / "13_광역버스통행시간_2024_종일_셀노선.csv"
TIME_BAND_CELL_PATH = OUT1 / "14_시간대별_승차인원_셀.csv"
TRAVEL_TIME_2024_CELL_PATH = OUT1 / "15_광역버스통행시간_2024_종일_셀.csv"

# analysis에서 생성하는 지도·표 등은 outputs에 저장
OUT2 = PROJECT_DIR / "outputs"
ROUTE_LEVEL_PATH = OUT2 / "route_level_change.csv"
MAP_PATH = OUT2 / "map_change_rate.html"

for folder in (OUT1, OUT2, PROCESSED_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# =========================================================
# 분석 설정
# =========================================================
MORNING_HOURS = (7, 9)  # 07:00~09:59
ALL_DAY_HOURS = (0, 23)
GYEONGGI, SEOUL = 41, 11
# GTX-A와 목적지가 겹치는 서울 측 역: 하차 정류장이 1km 이내면 'GTX 경쟁 목적지'
COMPETING_STATIONS = {"연신내": (37.618855, 126.920859), "서울역": (37.555980, 126.972091)}
COMPETING_RADIUS_M = 1000
# GTX-A역 좌표: 경훈 보행시간 파일(feature_02)에 쓰인 값과 동일하게 맞춤
GTX_STATIONS = {"운정중앙": (37.716670, 126.728330), "킨텍스": (37.665000, 126.748060), "대곡": (37.631626, 126.811024)}

# 본모형 독립변수 (09_회귀분석_격자통합데이터.csv 컬럼명)
X_MAIN = ["walk_min", "ic_access", "competing_share", "pop_20_50", "buildings"]
VARIABLE_LABELS = {
    "walk_min": "GTX역 보행시간(분)", "ic_access": "IC까지 도로거리(km)", "competing_share": "GTX 경쟁 목적지 비율(%)",
    "pop_20_50": "2024년 20~50대 인구", "pop_elderly": "2024년 고령인구", "buildings": "2024년 건축물 수",
    "board_2024": "2024년 07~09시 승차", "board_2024_all": "2024년 종일 승차", "log_board_2024": "log(1+2024년 07~09시 승차)",
    "walk_20": "보행 20분 이내", "walk_20_60": "보행 20~60분",
}


# =========================================================
# 공통 함수
# =========================================================
def setup_notebook():
    import warnings

    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore")
    plt.rcParams["font.family"] = "AppleGothic" if sys.platform == "darwin" else "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)


def read_csv(path):
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, str(path))


def load_model_data():
    """격자 단위 회귀용 데이터와 보행시간 구간 더미를 읽는다.

    최종모형의 GTX 경쟁 목적지 비율은 셀×노선 변수이므로
    load_cell_route_all_day()에서 별도로 결합한다.
    """
    data = read_csv(MODEL_DATA_PATH)
    data["셀 ID"] = pd.to_numeric(data["셀 ID"], errors="coerce").astype("Int64")
    data = data.drop(columns=["competing_share", "competing_share_missing"], errors="ignore")
    data["log_board_2024"] = np.log1p(data["board_2024"])
    data["walk_20"] = (data["walk_min"] <= 20).astype(float)
    data["walk_20_60"] = ((data["walk_min"] > 20) & (data["walk_min"] <= 60)).astype(float)
    data["walk_group"] = pd.cut(data["walk_min"], [0, 20, 60, np.inf], labels=["20분 이내", "20~60분", "60분 초과"])
    return data


def load_cell_route_all_day():
    """최종 종일 모형용 셀×노선 집계자료를 읽는다.

    원시 교통카드 자료를 공개하지 않고도 Poisson QMLE와 NB2 최종모형을
    재현할 수 있도록 2024·2025 승차인원만 셀×노선 단위로 집계한 파일이다.
    """
    frame = read_csv(CELL_ROUTE_ALL_DAY_PATH)
    required = {"셀 ID", "노선", "y2024", "y2025"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"셀×노선 집계자료에 필요한 열이 없습니다: {sorted(missing)}")
    frame["셀 ID"] = pd.to_numeric(frame["셀 ID"], errors="coerce").astype("Int64")
    frame["노선"] = frame["노선"].astype(str)
    frame["y2024"] = pd.to_numeric(frame["y2024"], errors="coerce").fillna(0)
    frame["y2025"] = pd.to_numeric(frame["y2025"], errors="coerce").fillna(0)
    frame = frame.dropna(subset=["셀 ID"]).copy()

    competing = read_csv(COMPETING_CELL_ROUTE_PATH)
    competing["셀 ID"] = pd.to_numeric(competing["셀 ID"], errors="coerce").astype("Int64")
    competing["노선"] = competing["노선"].astype(str).str.removesuffix(".0")
    share_col = "GTX 경쟁 목적지 승차인원 비율(%)"
    competing = competing[["셀 ID", "노선", share_col]].rename(columns={share_col: "competing_share"})
    competing["competing_share"] = pd.to_numeric(competing["competing_share"], errors="coerce")
    frame = frame.merge(competing, on=["셀 ID", "노선"], how="left", validate="one_to_one")
    frame["competing_share_missing"] = frame["competing_share"].isna().astype(int)
    frame["competing_share"] = frame["competing_share"].fillna(0)
    return frame


def coef_table(result, rate=False):
    """계수표. rate=True면 exp(coef)-1 = 변화율 상대 차이(%)를 추가한다."""
    table = pd.DataFrame({"coef": result.params, "p": result.pvalues})
    table = table.drop(index="alpha", errors="ignore")
    if rate:
        table["변화율 차이(%)"] = (np.exp(table["coef"]) - 1) * 100
    table["유의"] = np.select([table.p < 0.01, table.p < 0.05, table.p < 0.1], ["***", "**", "*"], "")
    table.index = [VARIABLE_LABELS.get(name, name) for name in table.index]
    return table.round(4)


def rate_model(frame, y_after, y_before, x_cols, family="poisson"):
    """변화율 모형: y_after ~ X, offset = log(y_before). 2024년 0명 셀은 제외."""
    import statsmodels.api as sm

    frame = frame[frame[y_before] > 0]
    X = sm.add_constant(frame[x_cols])
    offset = np.log(frame[y_before])
    if family == "poisson":
        return sm.GLM(frame[y_after], X, family=sm.families.Poisson(), offset=offset).fit(cov_type="HC1")
    return sm.NegativeBinomial(frame[y_after], X, offset=offset).fit(disp=0, maxiter=500)


def stable_centers():
    """공급구조 유지 셀 132개의 중심점 (EPSG:5179)."""
    import geopandas as gpd

    stable = read_csv(STABLE_PATH)
    stable["셀 ID"] = pd.to_numeric(stable["셀 ID"], errors="coerce")
    stable = stable.dropna(subset=["셀 ID", "중심점 위도", "중심점 경도"]).drop_duplicates("셀 ID").copy()
    stable["셀 ID"] = stable["셀 ID"].astype(int)
    return gpd.GeoDataFrame(
        stable[["셀 ID", "중심점 위도", "중심점 경도"]].copy(),
        geometry=gpd.points_from_xy(stable["중심점 경도"], stable["중심점 위도"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5179")


def stable_grid_polygons():
    """공급구조 유지 셀의 250m 폴리곤.

    빈격자(250m)를 고양·파주 경계로 잘라 순번으로 셀 ID를 부여한다 (Analisis2와 같은 방식이어야 셀 ID가 맞는다).
    """
    import geopandas as gpd

    stable_ids = set(pd.to_numeric(read_csv(STABLE_PATH)["셀 ID"], errors="coerce").dropna().astype(int))
    grid_crs = gpd.read_file(GRID_PATH, rows=1).crs
    admin = gpd.read_file(ADMIN_PATH).to_crs(grid_crs)
    admin["sgg_code"] = admin["sgg"].astype(str).str.extract(r"([0-9]+)")[0].str.zfill(5)
    city = admin[admin["sgg_code"].str.startswith(("4128", "4148"))]
    boundary = city.geometry.union_all() if hasattr(city.geometry, "union_all") else city.geometry.unary_union
    grid = gpd.read_file(GRID_PATH, bbox=boundary.bounds)
    grid = grid[grid.geometry.intersects(boundary)].copy().reset_index(drop=True)
    grid["_geometry_key"] = grid.geometry.apply(lambda x: x.wkb if x is not None else None)
    grid = grid.drop_duplicates("_geometry_key").drop(columns="_geometry_key").reset_index(drop=True)
    grid["셀 ID"] = np.arange(1, len(grid) + 1)
    return grid[grid["셀 ID"].isin(stable_ids)][["셀 ID", "geometry"]].copy()


def haversine_m(lat, lon, station_lat, station_lon):
    radius = 6_371_008.8
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(station_lat), np.radians(station_lon)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def is_competing_destination(alight_lat, alight_lon):
    """하차 좌표가 연신내역·서울역 1km 이내인지 (좌표 결측은 False)."""
    distance = np.min(
        [haversine_m(np.asarray(alight_lat, float), np.asarray(alight_lon, float), lat, lon)
         for lat, lon in COMPETING_STATIONS.values()],
        axis=0,
    )
    return distance <= COMPETING_RADIUS_M


def stop_to_cell_map():
    """정류장 ID(str) → 셀 ID. 2024·2025 정류장 ID를 모두 매핑한다 (1_01과 같은 규칙)."""
    stable = read_csv(STABLE_PATH)
    mapping = {}
    for _, row in stable.iterrows():
        for column in ["2024년 정류장 ID", "2025년 정류장 ID"]:
            for stop_id in str(row[column]).split("|"):
                stop_id = stop_id.strip().removesuffix(".0")
                if stop_id and stop_id.lower() != "nan":
                    mapping[stop_id] = int(row["셀 ID"])
    return mapping


def load_trips():
    """교통카드 2개 연도를 한 표로: 연도, 노선, 정류장, 셀 ID(안정 셀이 아니면 NaN), 승차시각, 승차인원, 좌표."""
    stop_to_cell = stop_to_cell_map()
    frames = []
    for year, path in RAW_PATHS.items():
        raw = read_csv(path)
        stop = raw["승차정류장ID"].astype(str).str.removesuffix(".0")
        frames.append(pd.DataFrame({
            "연도": year,
            "노선": raw["query_route_no"].astype(str),
            "정류장": stop,
            "셀 ID": stop.map(stop_to_cell),
            "시": pd.to_datetime(raw["ride_dt"].astype(str), format="%Y%m%d%H%M%S", errors="coerce").dt.hour,
            "승차인원": pd.to_numeric(raw["utztn_nope"], errors="coerce").fillna(0),
            "위도": pd.to_numeric(raw["승차위도"], errors="coerce"),
            "경도": pd.to_numeric(raw["승차경도"], errors="coerce"),
        }))
    return pd.concat(frames, ignore_index=True)


def nearest_gtx_km(lat, lon):
    """최근접 GTX-A역까지 직선거리(km)."""
    return np.min([haversine_m(np.asarray(lat, float), np.asarray(lon, float), la, lo)
                   for la, lo in GTX_STATIONS.values()], axis=0) / 1000


def poisson_rate(y_after, y_before, X, cluster=None):
    """변화율 Poisson (offset=log y_before). cluster가 있으면 군집 강건 표준오차."""
    import statsmodels.api as sm

    model = sm.GLM(y_after, sm.add_constant(X), family=sm.families.Poisson(), offset=np.log(y_before))
    if cluster is None:
        return model.fit(cov_type="HC1")
    return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(cluster)[0]})
