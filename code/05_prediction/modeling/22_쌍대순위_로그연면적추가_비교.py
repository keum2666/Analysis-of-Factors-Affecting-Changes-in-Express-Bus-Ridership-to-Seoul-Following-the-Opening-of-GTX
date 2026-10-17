"""기존4변수와 log1p(연면적[천㎡]) 추가5변수 쌍대순위 비교.

같은126개/정답/외부LOSO/내부C선택을 유지한다. 기존 파일 수정 없음.
외부 결과로 후보/설정을 추가 조정하지 않는다. CSV/모델 파일 저장 없음.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent

def load_ranker(extra):
    spec=importlib.util.spec_from_file_location('pairwise_with_floor_'+str(extra),next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-8_A3_쌍대비교_순위모형.py')))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if extra:module.FEATURES=[*module.FEATURES,'log1p_floor_area']
    return module

def evaluate():
    parts=[];folds=[];tops=[]
    for extra,name in [(False,'기존4변수'),(True,'로그연면적추가5변수')]:
        r=load_ranker(extra);data=r.load_base_module().load_data()
        assert len(data)==126 and data['셀 ID'].is_unique
        assert data[r.FEATURES].notna().all().all() and np.isfinite(data[r.FEATURES].to_numpy()).all()
        assert np.allclose(data.log1p_floor_area,np.log1p(data['2024년 건축물 연면적(천㎡)']))
        for station in data.nearest_station.unique():
            train=data[data.nearest_station!=station]
            test=data[data.nearest_station==station].copy()
            assert set(train['셀 ID']).isdisjoint(test['셀 ID'])
            c,inner=r.choose_c_inner_loso(train)
            x,y=r.make_pairs(train);pipe=r.make_model(c).fit(x,y)
            test['점수']=r.cell_scores(pipe,test)
            test['Y']=100*(test.board_2025_all/test.board_2024_all-test.board_2025_all.sum()/test.board_2024_all.sum())
            assert ((test.Y>0).astype(int)==test.share_increase).all()
            loss,count=r.pairwise_log_loss(test,test['점수'].to_numpy())
            ordered=test.sort_values('점수',ascending=False)
            test['순위']=pd.Series(np.arange(1,len(test)+1),index=ordered.index).reindex(test.index)
            k=int(np.ceil(len(test)*.2));test['상위20']=test['순위']<=k
            selected=ordered.head(k)
            assert set(test.loc[test['상위20'],'셀 ID'])==set(selected['셀 ID'])
            precision=(selected.Y>0).mean()
            folds.append({'모형':name,'검증역':station,'n':len(test),'선택C':c,'내부AUC':inner,
                          'AUC':roc_auc_score(test.share_increase,test['점수']),
                          'PairwiseLoss':loss,'외부비교쌍수':count,'상위20수':k,
                          '상위20적중수':int((selected.Y>0).sum()),'Precision20':precision,
                          'Lift20':precision/(test.Y>0).mean(),'상위Y중앙값':selected.Y.median(),
                          '상위Y가중평균':np.average(selected.Y,weights=selected.board_2024_all),
                          '상위기초20미만수':int((selected.board_2024_all<20).sum())})
            for rate in [.1,.2,.3]:
                k_rate=int(np.ceil(len(test)*rate));sub=ordered.head(k_rate);p=(sub.Y>0).mean()
                tops.append({'모형':name,'검증역':station,'상위비율':rate,'K':k_rate,'적중수':int((sub.Y>0).sum()),
                             'Precision':p,'Lift':p/(test.Y>0).mean(),'실제Y중앙값':sub.Y.median()})
            test['모형']=name;parts.append(test)
    folds=pd.DataFrame(folds);oof=pd.concat(parts,ignore_index=True);overall=[]
    assert oof.groupby('모형')['셀 ID'].nunique().eq(126).all()
    for name,g in folds.groupby('모형',sort=False):
        top=oof[(oof['모형']==name)&oof['상위20']]
        overall.append({'모형':name,'역평균AUC':g.AUC.mean(),'격자수가중AUC':np.average(g.AUC,weights=g.n),
                        '전체쌍PairwiseLoss':np.average(g.PairwiseLoss,weights=g['외부비교쌍수']),
                        '상위27적중수':int((top.Y>0).sum()),'상위27Precision':(top.Y>0).mean(),
                        '상위Y중앙값':top.Y.median(),'상위기초20미만수':int((top.board_2024_all<20).sum())})
    overall=pd.DataFrame(overall)
    assert abs(overall.loc[overall['모형']=='기존4변수','역평균AUC'].iloc[0]-.7311)<.0001
    assert overall.loc[overall['모형']=='기존4변수','상위27적중수'].iloc[0]==22
    return oof,folds,overall,pd.DataFrame(tops)

def main():
    oof,folds,overall,tops=evaluate()
    print('추가 변수 = log(1+2024년 건축물 연면적[천㎡]); 보행시간은 기존 원 단위 유지.')
    for title,t in [('전체',overall),('역별',folds),('상위비율별',tops)]:
        print('\n=====',title,'=====');print(t.round(4).to_string(index=False))
    print('\n===== 5변수 상위27개 =====')
    print(oof[(oof['모형']=='로그연면적추가5변수')&oof['상위20']].sort_values(['nearest_station','순위'])[['nearest_station','셀 ID','순위','board_2024_all','board_2025_all','Y']].round(3).to_string(index=False))
    print('\n같은 외부 자료에서 후보를 비교한 탐색이며 최종 선택 성능/GTX-C 성능 입증이 아니다.')
    print('CSV/모델/그림 파일 저장 없음.')

if __name__=='__main__':main()
