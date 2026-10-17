"""기초40명 이상73개 대상5변수 쌍대순위 고정 실험. 23번은 보존한다.

원래126개 기준 권역변화율/비중/정답 유지. 외부LOSO/내부C선택.
같은73개에서 전체학습 모형과 비교. 결과/모델 파일 저장 없음.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pairwise_24_base',next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('23_쌍대순위_5변수_10명이상.py')))
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
STATIONS=prior.STATIONS

def evaluate():
    r=prior.base.load_ranker(True)
    original=r.load_base_module().load_data()
    assert len(original)==126 and original['셀 ID'].is_unique
    totals=original.groupby('nearest_station')[['board_2024_all','board_2025_all']].sum()
    original['R_s']=original.nearest_station.map(100*(totals.board_2025_all/totals.board_2024_all-1))
    original['R_i']=100*(original.board_2025_all/original.board_2024_all-1)
    original['Y']=original.R_i-original.R_s
    data=original[original.board_2024_all>=40].copy()
    assert len(data)==73 and data[r.FEATURES].notna().all().all()
    assert np.isfinite(data[r.FEATURES].to_numpy()).all()
    assert ((data.Y>0).astype(int)==data.share_increase).all()
    selected=[];settings=[];coverage=[]
    for station in STATIONS:
        train=data[data.nearest_station!=station]
        test=data[data.nearest_station==station].copy()
        assert set(train['셀 ID']).isdisjoint(test['셀 ID'])
        assert test.share_increase.nunique()==2
        c,inner=r.choose_c_inner_loso(train)
        x,y=r.make_pairs(train);model=r.make_model(c).fit(x,y)
        test['점수']=r.cell_scores(model,test);selected.append(test)
        settings.append({'검증역':station,'학습n':len(train),'검증n':len(test),
                         '학습양방향쌍수':len(y),'선택C':c,'내부역평균AUC':inner})
        region=original[original.nearest_station==station]
        coverage.append({'권역':station,'원래n':len(region),'40명이상n':len(test),'제외n':len(region)-len(test),
                         '유지된개통전승차비율':test.board_2024_all.sum()/region.board_2024_all.sum(),
                         '고정권역변화율':test.R_s.iloc[0]})
    new=pd.concat(selected,ignore_index=True)
    rows,tops,oof=prior.score_tables(new,'40명이상재학습',r)
    full,_,_,_=prior.base.evaluate()
    old=full[(full['모형']=='로그연면적추가5변수')&(full.board_2024_all>=40)].drop(columns=['모형','순위','상위20']).copy()
    old['R_s']=old.nearest_station.map(100*(totals.board_2025_all/totals.board_2024_all-1))
    old['R_i']=100*(old.board_2025_all/old.board_2024_all-1);old['Y']=old.R_i-old.R_s
    assert set(new['셀 ID'])==set(old['셀 ID'])
    assert np.allclose(new.set_index('셀 ID').sort_index().Y,old.set_index('셀 ID').sort_index().Y)
    oldrows,oldtops,oldoof=prior.score_tables(old,'전체학습_같은73개평가',r)
    rows=pd.concat([oldrows,rows],ignore_index=True);tops=pd.concat([oldtops,tops],ignore_index=True)
    overall=[]
    for name,g in rows.groupby('모형',sort=False):
        selected=(oldoof if name.startswith('전체') else oof).loc[lambda x:x['상위20']]
        overall.append({'모형':name,'평가n':g.n.sum(),'역평균AUC':g.AUC.mean(),
                        '격자수가중AUC':np.average(g.AUC,weights=g.n),
                        '전체쌍Loss':np.average(g.PairwiseLoss,weights=g['비교쌍수']),
                        '상위수':len(selected),'적중수':int((selected.Y>0).sum()),
                        '상위Precision':(selected.Y>0).mean(),'상위실제Y중앙값':selected.Y.median()})
    return oof,rows,pd.DataFrame(overall),tops,pd.DataFrame(settings),pd.DataFrame(coverage)

def main():
    oof,rows,overall,tops,settings,coverage=evaluate()
    for title,t in [('대상 범위',coverage),('학습 설정',settings),('공통73개 비교',overall),('역별',rows),('상위비율별',tops)]:
        print('\n=====',title,'=====');print(t.round(4).to_string(index=False))
    print('\n===== 상위16개 =====')
    print(oof[oof['상위20']].sort_values(['nearest_station','순위'])[['nearest_station','셀 ID','순위','board_2024_all','board_2025_all','R_s','R_i','Y']].round(3).to_string(index=False))
    print('\n40명은 이번 적용범위 기준이며 하루40명만으로 버스1대 공급유지 필요성을 판정하지 않는다.')
    print('126/102/73개는 평가대상이 다르다. 이번 결과로40명 기준이 최적이라고 선택하지 않는다.')
    print('C의 기초40명 미만은 적용 대상 아님/별도 검토로 표시. CSV/모델/그림 저장 없음.')

if __name__=='__main__':main()
