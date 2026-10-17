"""16번 노트북의 유사사례 탐색. 미래 수요 예측모형으로 해석하지 않는다."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# 기존 5변수 구성을 유지한다. 결과를 보고 변수나 K를 고르지 않는다.
FEATURES = ['board_2024_all', 'log1p_walk_min', 'competing_share',
            'bus_time_mean', 'log1p_floor_area']
LABELS = ['개통 전 승차인원(명)', 'log(1+보행시간[분])', '경쟁 목적지 비율(%)',
          '버스 평균 통행시간(분)', 'log(1+연면적[천㎡])']
K = 5

def features(frame, columns):
    x = frame[columns].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('비교 변수에 결측/무한값이 있습니다. 자료를 확인한 뒤 비교하세요.')
    return x

def neighbors(reference, query, columns=FEATURES, k=K):
    if len(reference) < k:
        raise ValueError('비교 사례가 K보다 적습니다.')
    scaler = StandardScaler().fit(features(reference, columns))
    xr = scaler.transform(features(reference, columns))
    xq = scaler.transform(features(query, columns))
    distance = pairwise_distances(xq, xr)/np.sqrt(len(columns))
    # 동률은 reference의 셀 ID 정렬 순서에 따른다.
    order = np.argsort(distance, axis=1, kind='stable')[:, :k]
    return order, np.take_along_axis(distance, order, axis=1), scaler

def reference_cutoff(reference, columns=FEATURES, k=K):
    """참조역들 사이의 K개 평균거리 95분위. 결과값을 사용하지 않는 경고선."""
    scaler = StandardScaler().fit(features(reference, columns))
    x = scaler.transform(features(reference, columns))
    ds = pairwise_distances(x,x)/np.sqrt(len(columns))
    stations = reference.nearest_station.to_numpy()
    ds[stations[:,None] == stations[None,:]] = np.inf
    if np.isfinite(ds).sum(axis=1).min() < k:
        raise ValueError('다른 역의 참조 사례가 부족합니다.')
    return float(np.quantile(np.sort(ds, axis=1)[:,:k].mean(axis=1), .95))

def compare(reference, query, columns=FEATURES, k=K, with_outcomes=True):
    r = reference.sort_values('셀 ID').reset_index(drop=True)
    q = query.reset_index(drop=True)
    idx, distances, _ = neighbors(r,q,columns,k)
    cutoff = reference_cutoff(r,columns,k)
    change = r.board_2025_all-r.board_2024_all
    rate = change.div(r.board_2024_all.where(r.board_2024_all>0))*100
    rows, links = [], []
    for i in range(len(q)):
        nn = r.iloc[idx[i]]
        delta = change.iloc[idx[i]]
        rates = rate.iloc[idx[i]]
        outside = [col for col in columns if q.iloc[i][col]<r[col].min() or q.iloc[i][col]>r[col].max()]
        row = {'셀 ID':q.iloc[i]['셀 ID'], '권역':q.iloc[i]['nearest_station'],
               '개통전승차':q.iloc[i]['board_2024_all'], '평균거리':distances[i].mean(),
               '거리경고선':cutoff,'먼사례주의':distances[i].mean()>cutoff,
               '범위밖변수':', '.join(outside), '사례역수':nn.nearest_station.nunique(),
               '사례변화중앙값':delta.median(), '사례변화최소':delta.min(),
               '사례변화최대':delta.max(), '사례변화IQR':delta.quantile(.75)-delta.quantile(.25),
               '사례변화율중앙값':rates.median(), '사례기초20미만수':int((nn.board_2024_all<20).sum())}
        if with_outcomes:
            actual = q.iloc[i].board_2025_all-q.iloc[i].board_2024_all
            row.update({'실제변화량':actual,'중앙값차이절대값':abs(actual-delta.median()),
                        '사례범위내':delta.min()<=actual<=delta.max()})
        rows.append(row)
        for rank,j in enumerate(idx[i],1):
            a = r.iloc[j]
            links.append({'대상셀':q.iloc[i]['셀 ID'],'대상권역':q.iloc[i]['nearest_station'],
                          '유사순서':rank,'사례셀':a['셀 ID'],'사례권역':a.nearest_station,
                          '거리':distances[i,rank-1],'개통전승차':a.board_2024_all,
                          '개통후승차':a.board_2025_all,'변화량':change.iloc[j],
                          '변화율(%)':rate.iloc[j], **{c:a[c] for c in columns}})
    return pd.DataFrame(rows), pd.DataFrame(links)

def validate(data):
    summaries, links = [], []
    for station in ['운정중앙','킨텍스','대곡']:
        ref = data[data.nearest_station!=station]
        qry = data[data.nearest_station==station]
        assert set(ref['셀 ID']).isdisjoint(qry['셀 ID'])
        s,l = compare(ref,qry)
        s['방식']='지역교통5변수'; summaries.append(s); links.append(l)
        s,_ = compare(ref,qry,['board_2024_all'])
        s['방식']='승차규모만'; summaries.append(s)
    s = pd.concat(summaries,ignore_index=True)
    l = pd.concat(links,ignore_index=True)
    assert len(s)==252 and len(l)==126*K
    assert (l['대상권역']!=l['사례권역']).all()
    return s,l

def apply_c(data_a, c_frame, definition_confirmed=False):
    """C 입력은 개통전 자료만. A 연도 이름의 열은 공통 입력 필드로 매핑한다."""
    if not definition_confirmed:
        raise ValueError('C의 기준연도·단위·목적지 정의·보행 기준을 A와 비교해 확인해야 합니다.')
    required = ['셀 ID','nearest_station',*FEATURES]
    missing = set(required)-set(c_frame.columns)
    if missing:
        raise ValueError(f'C 필수 열 누락: {sorted(missing)}')
    if c_frame.empty or not c_frame['셀 ID'].is_unique:
        raise ValueError('C 자료가 비어 있거나 셀 ID가 중복됩니다.')
    return compare(data_a,c_frame[required],with_outcomes=False)
