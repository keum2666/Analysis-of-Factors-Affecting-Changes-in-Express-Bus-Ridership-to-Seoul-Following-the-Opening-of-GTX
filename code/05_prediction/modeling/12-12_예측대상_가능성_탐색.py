"""현재 126개 격자에서 예측 가능한 대상의 탐색적 비교. 결과 파일 저장 없음.

사전에 정한 대상: 개통 후 승차인원, 개통 후 비중, 비중 변화폭,
개통 후 일일 40명 이상 잔존수요 여부. 40명은 서비스 유지 타당성 기준이
아니라 이전에 제안한 규모 기준을 탐색하는 예시이다.
외부 LOSO의 검증역은 전처리·규제·후보 선택에 사용하지 않는다.
후보별 규제와 후보 전체 선택 모두 학습역 내부 LOSO에서 수행한다.
대상 자체를 이번 결과로 선택하면 탐색이므로 독립 검증을 대체하지 않는다.
"""
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
    average_precision_score, balanced_accuracy_score, brier_score_loss, log_loss,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
STATIONS = ['대곡', '운정중앙', '킨텍스']
ALPHAS = np.logspace(-3, 3, 13)
A3 = ['pre_share_pct', 'log1p_board_2024', 'walk_min', 'competing_share']
FULL = A3 + ['bus_time_mean', 'ic_access', 'log1p_pop_20_50', 'log1p_floor_area', '주거지역 비율(%)']


def load_data():
    spec = importlib.util.spec_from_file_location('base_12_12', next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-2_Ridge_Logistic_예측성능_개선실험.py')))
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    data = base.load_data().copy()
    assert len(data) == 126 and data['셀 ID'].nunique() == 126
    assert data[['board_2024_all', 'board_2025_all']].notna().all().all()
    data['post_pct'] = data['share_post'] * 100
    data['delta_pct'] = data['post_pct'] - data['pre_share_pct']
    data['post40'] = (data['board_2025_all'] >= 40).astype(int)
    return data


def specs_for(task):
    if task == 'post40':
        return [('기초수요단독_Logistic', 'logistic', ['log1p_board_2024']),
                ('A3_Logistic', 'logistic', A3), ('전체9_Logistic', 'logistic', FULL)]
    if task == 'delta_pct':
        return [('비중불변', 'same', []), ('기초비중_Ridge', 'direct', ['pre_share_pct']),
                ('A3_Ridge', 'direct', A3), ('전체9_Ridge', 'direct', FULL)]
    specs = [('개통전유지', 'same', []), ('학습역총량비례', 'ratio', [])]
    for label, columns in [('기초수요', ['log1p_board_2024']), ('A3', A3), ('전체9', FULL)]:
        raw = ['board_2024_all' if c == 'log1p_board_2024' else c for c in columns]
        specs.extend([(label + '_원단위Ridge', 'raw', raw), (label + '_로그Ridge', 'log', columns)])
    if task == 'post_pct':
        specs.extend([('A3_비중직접Ridge', 'direct', A3), ('전체9_비중직접Ridge', 'direct', FULL)])
    return specs


def target(frame, task):
    return frame[task].to_numpy(float)


def normalize_share(values, test):
    out = np.zeros(len(test))
    groups = test['nearest_station'].to_numpy()
    for station in np.unique(groups):
        idx = groups == station
        positive = np.maximum(values[idx], 0)
        out[idx] = positive / positive.sum() * 100 if positive.sum() > 0 else test.loc[idx, 'pre_share_pct']
    return out


def predict(train, test, task, spec, strength):
    _, mode, columns = spec
    if mode == 'same':
        if task == 'delta_pct': return np.zeros(len(test))
        if task == 'post_pct': return test['pre_share_pct'].to_numpy(float)
        return test['board_2024_all'].to_numpy(float)
    if mode == 'ratio':
        factor = train['board_2025_all'].sum() / train['board_2024_all'].sum()
        values = test['board_2024_all'].to_numpy(float) * factor
    else:
        if mode == 'logistic':
            estimator = LogisticRegression(C=strength, penalty='l2', solver='liblinear', max_iter=10000, random_state=42)
            y_train = target(train, task)
        else:
            estimator = Ridge(alpha=strength)
            y_train = target(train, task) if mode == 'direct' else train['board_2025_all'].to_numpy(float)
            if mode == 'log': y_train = np.log1p(y_train)
        pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
        pipeline.fit(train[columns], y_train)
        if mode == 'logistic': return pipeline.predict_proba(test[columns])[:, 1]
        values = pipeline.predict(test[columns])
        if mode == 'log': values = np.expm1(values)  # 원 단위 MAE로 평가하며, 원 단위 평균의 불편추정치는 아니다.
    if task == 'delta_pct':
        return normalize_share(test['pre_share_pct'].to_numpy(float) + values, test) - test['pre_share_pct'].to_numpy(float)
    if task == 'post_pct': return normalize_share(values, test)
    return np.maximum(values, 0)


def inner_select(train, task, specs):
    selected = []
    for spec in specs:
        strengths = [1.] if spec[1] in ('same', 'ratio') else ALPHAS
        scored = []
        for strength in strengths:
            losses = []
            for station in train['nearest_station'].unique():
                fit = train[train['nearest_station'] != station]
                valid = train[train['nearest_station'] == station]
                pred = predict(fit, valid, task, spec, strength)
                loss = log_loss(target(valid, task), pred, labels=[0, 1]) if task == 'post40' else mean_absolute_error(target(valid, task), pred)
                losses.append(loss)
            scored.append((float(np.mean(losses)), strength))
        # 동률이면 강한 규제: Ridge는 큰 alpha, Logistic은 작은 C.
        best = min(scored, key=lambda z: (z[0], z[1] if spec[1] == 'logistic' else -z[1]))
        selected.append((spec, best[1], best[0]))
    return selected


def metrics(test, pred, task):
    y = target(test, task)
    if task == 'post40':
        return {'AUC': roc_auc_score(y, pred), 'AP': average_precision_score(y, pred),
                'BalancedAcc': balanced_accuracy_score(y, pred >= .5), 'Brier': brier_score_loss(y, pred)}
    out = {'MAE': mean_absolute_error(y, pred), 'RMSE': np.sqrt(mean_squared_error(y, pred)), 'R2': r2_score(y, pred)}
    if np.ptp(pred) > 0:
        out['Spearman'] = spearmanr(y, pred).statistic
    else:
        out['Spearman'] = np.nan
    k = int(np.ceil(len(test) * .2))
    ids = np.argsort(-pred, kind='stable')[:k]
    if task == 'board_2025_all':
        actual_top = set(np.argsort(-y, kind='stable')[:k])
        out['Top20_실제상위적중률'] = len(actual_top.intersection(ids)) / k
        out['Top20_잔존수요포착률'] = y[ids].sum() / y.sum()
        out['총량오차율'] = (pred.sum() / y.sum() - 1) * 100
    if task == 'delta_pct':
        # 비중불변 예측은 모두 동점이므로 상위20% 순위 평가를 하지 않는다.
        out['Top20_실제비중변화_pp'] = y[ids].mean() if np.ptp(pred) > 0 else np.nan
    return out


def evaluate(task, data):
    rows = []
    predictions = []
    specs = specs_for(task)
    for station in STATIONS:
        train = data[data['nearest_station'] != station]
        test = data[data['nearest_station'] == station]
        selected = inner_select(train, task, specs)
        winner = min(selected, key=lambda z: z[2])
        multi_candidates = [x for x in selected if x[0][0].startswith(('A3_', '전체9_'))]
        multi_winner = min(multi_candidates, key=lambda z: z[2]) if multi_candidates else None
        for spec, strength, inner_loss in selected:
            pred = predict(train, test, task, spec, strength)
            row = {'대상': task, '모형': spec[0], '검증역': station, 'n': len(test), '규제값': strength,
                   '내부손실': inner_loss, **metrics(test, pred, task)}
            rows.append(row)
            prediction = pd.DataFrame({'셀 ID': test['셀 ID'].to_numpy(), '검증역': station,
                                       '모형': spec[0], 'actual': target(test, task), 'prediction': pred})
            predictions.append(prediction)
            if spec[0] == winner[0][0]:
                rows.append({**row, '모형': '내부선택절차', '선택후보': spec[0]})
                predictions.append(prediction.assign(모형='내부선택절차'))
            if multi_winner is not None and spec[0] == multi_winner[0][0]:
                rows.append({**row, '모형': '다변수내부선택절차', '선택후보': spec[0]})
                predictions.append(prediction.assign(모형='다변수내부선택절차'))
    result = pd.DataFrame(rows)
    pred = pd.concat(predictions, ignore_index=True)
    cols = ['AUC', 'AP', 'BalancedAcc', 'Brier'] if task == 'post40' else ['MAE','RMSE','R2','Spearman']
    cols += [x for x in ['Top20_실제상위적중률','Top20_잔존수요포착률','Top20_실제비중변화_pp'] if x in result]
    macro = result.groupby('모형', sort=False)[cols].mean()
    if task != 'post40':
        for name, part in pred.groupby('모형'):
            macro.loc[name, '전체OOF_MAE'] = mean_absolute_error(part.actual, part.prediction)
            macro.loc[name, '전체OOF_R2'] = r2_score(part.actual, part.prediction)
            if task == 'board_2025_all':
                macro.loc[name, '전체OOF_WAPE'] = np.abs(part.actual - part.prediction).sum() / part.actual.sum()
    print(f'\n===== 대상: {task} | 역별 지표 단순평균 =====')
    print(macro.round(4).to_string())
    print('\n----- 검증역별 상세 -----')
    print(result[['모형','검증역','n',*cols,'규제값']].round(4).to_string(index=False))
    print('\n----- 내부 선택 후보 -----')
    print(result.loc[result['모형'].isin(['내부선택절차', '다변수내부선택절차']),
                     ['검증역','모형','선택후보','내부손실']].round(4).to_string(index=False))
    if task == 'post40':
        print('\n----- 전체 OOF 혼동행렬 [실제 0/1 행, 예측 0/1 열] -----')
        for name, part in pred.groupby('모형', sort=False):
            print(name, confusion_matrix(part.actual, part.prediction >= .5, labels=[0, 1]).tolist())
    return result, pred


def main():
    data = load_data()
    print('126개 동일 표본. 모든 전처리·규제·모형 선택은 외부 학습역 내부에서 수행.')
    print('범위: 개통 후 승차량 / 비중 / 비중변화 / 개통 후 40명 이상 여부')
    print(data.groupby('nearest_station').agg(n=('셀 ID','size'), 개통후40이상=('post40','sum'),
                                            개통전승차=('board_2024_all','sum'),개통후승차=('board_2025_all','sum')).to_string())
    for task in ['board_2025_all','post_pct','delta_pct','post40']:
        evaluate(task, data)
    print('\n주의: 대상 자체를 이번 결과로 고르면 탐색적 결과입니다. 독립된 역은 3개뿐입니다.')
    print('높은 개통후 규모 예측력은 기존 수요규모 지속성을 반영할 수 있으며 GTX의 효과가 아닙니다.')
    print('40명/일은 차량 한 대 운행의 필요성을 입증하는 기준이 아닙니다.')
    print('결과 파일은 저장하지 않습니다.')


if __name__ == '__main__':
    main()
