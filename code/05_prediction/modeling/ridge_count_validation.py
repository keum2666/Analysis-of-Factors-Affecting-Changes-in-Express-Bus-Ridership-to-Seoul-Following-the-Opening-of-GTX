"""011-2의 승차인원 회귀 비교. 외부 검증역은 alpha 선택/전처리에서 제외."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ridge_count_base', next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-12_예측대상_가능성_탐색.py')))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ALPHAS = np.logspace(-3, 3, 13)
STATIONS = ['운정중앙', '킨텍스', '대곡']
PRIMARY = '승차원단위_보행로그'
MODELS = {}
for xlog in [False, True]:
    for ylog in [False, True]:
        # 사용자가 지정한 OLS와의 변환 일관성: 보행시간은 로그로 고정.
        for wlog in [True]:
            label = ('승차원단위' if not xlog and not ylog else
                     '승차입력로그' if xlog and not ylog else
                     '승차정답로그' if ylog and not xlog else '승차양쪽로그')
            label += '_보행로그' if wlog else '_보행원단위'
            MODELS[label] = {'xlog': xlog, 'ylog': ylog, 'wlog': wlog,
                'features': ['log1p_board_2024' if xlog else 'board_2024_all',
                             'log1p_walk_min' if wlog else 'walk_min',
                             'competing_share', 'bus_time_mean', 'log1p_floor_area']}
MODELS['개통전승차제외_보행로그'] = {
    'xlog': False, 'ylog': False, 'wlog': True,
    'features': ['log1p_walk_min', 'competing_share', 'bus_time_mean', 'log1p_floor_area'],
}


def load_data():
    data = base.load_data().reset_index(drop=True)
    assert len(data) == 126 and data['셀 ID'].nunique() == 126
    assert (data[['board_2024_all', 'board_2025_all']] >= 0).all().all()
    return data


def predict(train, test, model, alpha):
    settings = MODELS[model]
    y = train.board_2025_all.to_numpy(float)
    if settings['ylog']:
        y = np.log1p(y)
    pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=alpha))
    pipe.fit(train[settings['features']], y)
    raw = pipe.predict(test[settings['features']])
    if settings['ylog']:
        raw = np.expm1(raw)
    # 음수 불가능한 승차인원을 위해 모든 모형에 동일한 사전 처리.
    return np.maximum(raw, 0), int((raw < 0).sum())


def choose_alpha(train, model):
    losses = []
    for alpha in ALPHAS:
        fold_losses = []
        for station in train.nearest_station.unique():
            fit = train[train.nearest_station != station]
            valid = train[train.nearest_station == station]
            p, _ = predict(fit, valid, model, alpha)
            fold_losses.append(mean_absolute_error(valid.board_2025_all, p))
        losses.append(np.mean(fold_losses))
    chosen = min(range(len(ALPHAS)), key=lambda i: (losses[i], -ALPHAS[i]))
    return ALPHAS[chosen], losses[chosen]


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {'MAE(명)': mean_absolute_error(y, p),
            'RMSE(명)': np.sqrt(mean_squared_error(y, p)), 'R²': r2_score(y, p),
            '평균오차(예측-실제)': np.mean(p-y),
            'WAPE(%)': 100*np.abs(p-y).sum()/y.sum() if y.sum() else np.nan}


def evaluate(data):
    parts, folds = [], []
    for station in STATIONS:
        train = data[data.nearest_station != station]
        valid = data[data.nearest_station == station]
        for model in [*MODELS, '총량비례 기준']:
            if model == '총량비례 기준':
                alpha, loss, negative = np.nan, np.nan, 0
                pred = valid.board_2024_all.to_numpy(float)*train.board_2025_all.sum()/train.board_2024_all.sum()
            else:
                alpha, loss = choose_alpha(train, model)
                pred, negative = predict(train, valid, model, alpha)
            part = valid[['셀 ID', 'nearest_station', 'board_2024_all', 'board_2025_all']].copy()
            part['모형'] = model
            part['예측승차인원'] = pred
            part['오차'] = pred - part.board_2025_all
            part['절대오차'] = part['오차'].abs()
            parts.append(part)
            folds.append({'모형': model, '검증역': station, 'n': len(valid), 'alpha': alpha,
                          '내부MAE': loss, '음수예측_0처리수': negative,
                          **metrics(part.board_2025_all, pred)})
    oof = pd.concat(parts, ignore_index=True)
    assert oof.groupby(['모형', '셀 ID']).size().eq(1).all()
    overall = pd.DataFrame([{'모형': model, **metrics(p.board_2025_all, p['예측승차인원'])}
                            for model, p in oof.groupby('모형', sort=False)])
    return overall, pd.DataFrame(folds), oof


def attach_geometry(data):
    import geopandas as gpd
    root = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
    grid_path = next((root/'data' / 'raw' / 'common' / 'grid' / '빈격자(250m)').glob('*.shp'))
    admin = gpd.read_file(root/'data' / 'raw' / 'common' / 'api_cache_gtx_a_routes'/'HangJeongDong_ver20260701.geojson')
    crs = gpd.read_file(grid_path, rows=1).crs
    admin = admin.to_crs(crs)
    codes = admin.sgg.astype(str).str.extract(r'([0-9]+)')[0].str.zfill(5)
    cities = admin[codes.str.startswith(('4128', '4148'))]
    boundary = cities.geometry.union_all() if hasattr(cities.geometry, 'union_all') else cities.geometry.unary_union
    grid = gpd.read_file(grid_path, bbox=boundary.bounds)
    grid = grid[grid.geometry.intersects(boundary)].copy()
    grid['_geometry_key'] = grid.geometry.apply(lambda g: g.wkb if g is not None else None)
    grid = grid.drop_duplicates('_geometry_key').reset_index(drop=True)
    grid['셀 ID'] = np.arange(1, len(grid)+1)
    result = grid[['셀 ID', 'geometry']].merge(data, on='셀 ID', validate='one_to_one')
    assert len(result) == 126 and result.geometry.notna().all()
    centers = gpd.GeoSeries(result.geometry.centroid, crs=result.crs).to_crs(4326)
    result['위도'], result['경도'] = centers.y, centers.x
    return result


def tag_errors(frame):
    frame = frame.copy()
    # 동률은 셀 ID 순서로 정렬; 절대오차가 작다는 이유만으로 상대오차까지 작다고 해석하지 않는다.
    for label, ascending in [('작은오차순위', True), ('큰오차순위', False)]:
        ordered = frame.sort_values(['nearest_station', '절대오차', '셀 ID'], ascending=[True, ascending, True])
        ranks = ordered.groupby('nearest_station').cumcount()+1
        frame[label] = ranks.reindex(frame.index)
    return frame


if __name__ == '__main__':
    overall, folds, _ = evaluate(load_data())
    print(overall.round(4).to_string(index=False))
    print(folds[folds.모형.isin([PRIMARY, '승차양쪽로그_보행로그'])].round(4).to_string(index=False))
