from pathlib import Path
import hashlib, json, os, re, time
from typing import Any

import pandas as pd
import requests
import geopandas as gpd

# 프로젝트 루트(UROP): config.py와 code/가 있는 상위 폴더
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ROOT / '.env')
except ImportError:
    env = ROOT / '.env'
    if env.exists():
        for line in env.read_text(encoding='utf-8-sig').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# 실행 위치가 notebook/kernel 기준으로 달라도 파일 옆 .env를 다시 확인한다.
if not os.getenv('DATA_GO_KR_SERVICE_KEY'):
    env = ROOT / '.env'
    if env.exists():
        for line in env.read_text(encoding='utf-8-sig').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

SERVICE_KEY = os.getenv('DATA_GO_KR_SERVICE_KEY', '').strip()
if not SERVICE_KEY:
    raise RuntimeError('.env에 DATA_GO_KR_SERVICE_KEY가 없습니다.')

BASE_URL = 'https://apis.data.go.kr/1613000/BusRouteInfoInqireService'
CACHE_DIR, OUT_DIR = ROOT / 'data' / 'raw' / 'common' / 'api_cache_gtx_a_routes', ROOT / 'data' / 'raw' / 'gtx_a' / 'route_selection'
BOUNDARY_FILE = CACHE_DIR / 'HangJeongDong_ver20260701.geojson'
BOUNDARY_URL = 'https://raw.githubusercontent.com/vuski/admdongkor/master/ver20260701/HangJeongDong_ver20260701.geojson'
CACHE_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
PAGE_SIZE, RETRIES, TIMEOUT = 1000, 3, 30

TARGETS = [
    {'GTX_A_역':'운정중앙역','대상지역':'운정·파주','city_keywords':['파주'],'adm_patterns':['파주시']},
    {'GTX_A_역':'킨텍스역','대상지역':'일산서구','city_keywords':['고양'],'adm_patterns':['고양시일산서구']},
    {'GTX_A_역':'대곡역','대상지역':'덕양구','city_keywords':['고양'],'adm_patterns':['고양시덕양구']},
    {'GTX_A_역':'성남역','대상지역':'성남·분당','city_keywords':['성남'],'adm_patterns':['성남시분당구']},
    {'GTX_A_역':'구성역','대상지역':'용인·기흥구','city_keywords':['용인'],'adm_patterns':['용인시기흥구']},
    {'GTX_A_역':'동탄역','대상지역':'동탄·화성','city_keywords':['화성'],'adm_patterns':['동탄구']},
]

def s(v: Any) -> str:
    return '' if v is None else str(v).strip()

def val(row: dict, keys: list[str]) -> str:
    for key in keys:
        if key in row and s(row[key]): return s(row[key])
    return ''

def rows(payload: dict):
    response = payload.get('response', payload)
    header = response.get('header', {}) if isinstance(response, dict) else {}
    body = response.get('body', {}) if isinstance(response, dict) else {}
    items = body.get('items', {}) if isinstance(body, dict) else {}
    items = items.get('item', []) if isinstance(items, dict) else items
    if isinstance(items, dict): items = [items]
    return (items if isinstance(items, list) else []), {'header': header, 'body': body}

def cache_path(endpoint, params):
    public = {k:v for k,v in params.items() if k != 'serviceKey'}
    token = hashlib.sha256(json.dumps([endpoint, public], sort_keys=True).encode()).hexdigest()
    return CACHE_DIR / f'{endpoint}_{token}.json'

def get(endpoint: str, params: dict, cache=True):
    params = {'serviceKey': SERVICE_KEY, '_type': 'json', **params}
    path = cache_path(endpoint, params)
    if cache and path.exists(): return rows(json.loads(path.read_text(encoding='utf-8')))
    for attempt in range(RETRIES):
        try:
            r = requests.get(f'{BASE_URL}/{endpoint}', params=params, timeout=TIMEOUT)
            r.raise_for_status(); payload = r.json(); result, meta = rows(payload)
            code = val(meta['header'], ['resultCode','resultcode'])
            if code not in ('', '00', '0'): raise RuntimeError(f'resultCode={code}: {meta["header"]}')
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            return result, meta
        except Exception as exc:
            if attempt == RETRIES - 1:
                print(f'[API 실패] {endpoint}: {exc}'); return [], {'header':{},'body':{}}
            time.sleep(2 ** attempt)
    return [], {'header':{},'body':{}}

def paged(endpoint, params):
    result, page = [], 1
    while True:
        batch, meta = get(endpoint, {**params, 'pageNo':page, 'numOfRows':PAGE_SIZE})
        result.extend(batch)
        total = int(val(meta.get('body', {}), ['totalCount','totalcount']) or 0)
        if not batch or (total and len(result) >= total) or len(batch) < PAGE_SIZE: break
        page += 1
    return result

city_rows = paged('getCtyCodeList', {})
print('getCtyCodeList 실제 필드:', sorted(city_rows[0]) if city_rows else '응답 없음')
city_map = {val(x,['citycode','cityCode','city_code']):val(x,['cityname','cityName','city_name']) for x in city_rows}
city_map = {k:v for k,v in city_map.items() if k and v}

def city_codes(keywords): return [c for c,n in city_map.items() if any(k in n for k in keywords)]

def route_candidates(city_code):
    direct = paged('getRouteNoList', {'cityCode':city_code})
    if direct: return direct
    # routeNo가 필수인 API 버전의 fallback. 숫자 0~9와 광역 계열 prefix를 모두 조회한다.
    seen, result = set(), []
    for term in list('0123456789') + ['M','G','P']:
        for x in paged('getRouteNoList', {'cityCode':city_code, 'routeNo':term}):
            rid = val(x,['routeid','routeId','route_id'])
            if rid and rid not in seen: seen.add(rid); result.append(x)
    return result

def metropolitan(x):
    no, typ = val(x,['routeno','routeNo','route_no']), val(x,['routetp','routeType','route_tp'])
    text = f'{no} {typ}'.upper()
    return any(k in text for k in ['광역버스','광역급행','직행좌석','M버스'])

# 노선 등록 cityCode와 실제 운행지역이 다를 수 있으므로 경기도 전체를 조회한다.
# 예: M5107은 용인권을 운행하지만 API상 수원시 노선으로 등록될 수 있다.
# TAGO cityCode는 cityname에 '경기도'를 붙이지 않고 제공하는 경우가 있다.
# 경기도 시·군 표준 cityCode 대역(31xxx)을 사용해 등록 지자체 누락을 막는다.
gyeonggi_codes = [code for code in city_map if code.startswith(('310','311','312','313'))]
candidate_rows = []
for code in gyeonggi_codes:
    for x in route_candidates(code):
        y = dict(x); y.update(_citycode=code, _cityname=city_map[code])
        candidate_rows.append(y)
print('노선번호 조회 실제 필드:', sorted(candidate_rows[0]) if candidate_rows else '응답 없음')

def load_boundaries():
    if not BOUNDARY_FILE.exists():
        r = requests.get(BOUNDARY_URL, timeout=60); r.raise_for_status()
        BOUNDARY_FILE.write_bytes(r.content)
    return gpd.read_file(BOUNDARY_FILE).to_crs('EPSG:4326')

boundaries = load_boundaries()
print('행정동 경계 필드:', sorted(boundaries.columns))

def stop_points(stops):
    valid = []
    for i, row in enumerate(stops):
        stop_name = val(row, ['nodenm','nodeNm','node_name','nodeName'])
        if '미정차' in stop_name:
            continue
        try:
            lat, lon = float(val(row,['gpslati','gpsLati','latitude'])), float(val(row,['gpslong','gpsLong','longitude']))
            valid.append((i, row, lon, lat))
        except (TypeError, ValueError):
            continue
    return valid

def matched_dongs(stops):
    pts = stop_points(stops)
    if not pts: return {}
    points = gpd.GeoDataFrame({'stop_index':[x[0] for x in pts]}, geometry=gpd.points_from_xy([x[2] for x in pts], [x[3] for x in pts]), crs='EPSG:4326')
    joined = gpd.sjoin(points, boundaries[['adm_cd2','adm_nm','geometry']], how='left', predicate='within')
    return {int(r.stop_index): r for _, r in joined.dropna(subset=['adm_cd2']).iterrows()}

records, seen, info_fields, stop_fields = [], set(), None, None
for candidate in candidate_rows:
    if not metropolitan(candidate): continue
    route_id, city_code = val(candidate,['routeid','routeId','route_id']), candidate['_citycode']
    if not route_id or (city_code,route_id) in seen: continue
    seen.add((city_code,route_id))
    info = paged('getRouteInfoIem', {'cityCode':city_code,'routeId':route_id})
    stops = paged('getRouteAcctoThrghSttnList', {'cityCode':city_code,'routeId':route_id})
    if info and info_fields is None: info_fields = sorted(info[0])
    if stops and stop_fields is None: stop_fields = sorted(stops[0])
    dong_map = matched_dongs(stops)
    seoul_stops = [x for i, x in enumerate(stops) if i in dong_map and s(dong_map[i]['adm_nm']).startswith('서울특별시')]
    base = info[0] if info else candidate
    if not seoul_stops: continue
    seoul_order = min(int(val(x,['nodeord','nodeOrd','node_order']) or 999999) for x in seoul_stops)
    for target in TARGETS:
        region_stops = [x for i, x in enumerate(stops) if i in dong_map and any(p in s(dong_map[i]['adm_nm']) for p in target['adm_patterns'])]
        if not region_stops: continue
        target_order = min(int(val(x,['nodeord','nodeOrd','node_order']) or 999999) for x in region_stops)
        if target_order >= seoul_order: continue
        records.append({'GTX_A_역':target['GTX_A_역'],'대상지역':target['대상지역'],
            '버스번호':val(base,['routeno','routeNo','route_no']) or val(candidate,['routeno','routeNo','route_no']),
            '노선ID':route_id,'노선유형':val(base,['routetp','routeType','route_tp']) or val(candidate,['routetp','routeType','route_tp']),
            '기점':val(base,['startnodenm','startNodeNm','start_node_name']),'종점':val(base,['endnodenm','endNodeNm','end_node_name']),'서울진입여부':'Y'})

columns = ['GTX_A_역','대상지역','버스번호','노선ID','노선유형','기점','종점','서울진입여부']
result_df = pd.DataFrame(records, columns=columns).drop_duplicates(['GTX_A_역','노선ID']).sort_values(['GTX_A_역','버스번호'])
summary = result_df.groupby(['GTX_A_역','대상지역'])['버스번호'].apply(lambda x:', '.join(sorted(set(map(str,x))))).reset_index(name='서울방향_광역버스_후보') if not result_df.empty else pd.DataFrame(columns=['GTX_A_역','대상지역','서울방향_광역버스_후보'])
csv_df = result_df.copy()
# 표준 CSV에는 셀 서식이 없으므로 Excel의 날짜 자동변환을 막기 위한 표시용 접두사다.
csv_df['버스번호'] = csv_df['버스번호'].map(lambda x: '' if pd.isna(x) else "'" + str(x))
csv_df.to_csv(OUT_DIR/'gtx_a_seoul_metropolitan_bus_candidates.csv', index=False, encoding='utf-8-sig')
summary.to_csv(OUT_DIR/'gtx_a_seoul_metropolitan_bus_summary.csv', index=False, encoding='utf-8-sig')
try:
    # Excel이 7007-1을 날짜(7007-01-01)로 자동 변환하지 않도록 텍스트 서식을 고정한다.
    with pd.ExcelWriter(OUT_DIR/'gtx_a_seoul_metropolitan_bus_candidates.xlsx', engine='openpyxl') as writer:
        result_df.to_excel(writer, index=False, sheet_name='후보노선')
        summary.to_excel(writer, index=False, sheet_name='역별요약')
        for ws in writer.book.worksheets:
            for cell in ws[1]:
                if cell.value in ('버스번호', '노선ID'):
                    for data_cell in ws[cell.column_letter][1:]:
                        data_cell.number_format = '@'
                        data_cell.value = '' if data_cell.value is None else str(data_cell.value)
except ImportError:
    print('xlsx 저장을 위해 openpyxl 설치가 필요합니다.')
print('노선정보 실제 필드:', info_fields or '응답 없음'); print('경유정류소 실제 필드:', stop_fields or '응답 없음')
for t in TARGETS: print(f"{t['GTX_A_역']}: 고유 후보 노선 {(result_df['GTX_A_역']==t['GTX_A_역']).sum() if not result_df.empty else 0}개")
missing = [t['GTX_A_역'] for t in TARGETS if result_df.empty or not (result_df['GTX_A_역']==t['GTX_A_역']).any()]
print('검색되지 않은 역:', ', '.join(missing) if missing else '없음')
print('저장:', OUT_DIR/'gtx_a_seoul_metropolitan_bus_candidates.csv'); print('저장:', OUT_DIR/'gtx_a_seoul_metropolitan_bus_summary.csv')
print(result_df.to_string(index=False)); print(summary.to_string(index=False))
