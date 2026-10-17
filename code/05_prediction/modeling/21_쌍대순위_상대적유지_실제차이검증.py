"""기존 쌍대비교 OOF 순위를 재현해 상위20%의 실제 Y 크기를 진단한다.

새로운 학습목표/변수/하이퍼파라미터/상위비율을 탐색하지 않는다.
기존 12-8 절차로 재현하는 것이며 새로운 독립 검증이 아니다.
Y는 사후 진단에만 사용하며 점수/순위/상위격자 선택에 사용하지 않는다.
CSV/모델/그림 파일 저장 없음.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('original_pairwise',next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-8_A3_쌍대비교_순위모형.py')))
ranker=importlib.util.module_from_spec(spec);spec.loader.exec_module(ranker)
EXPECTED={14004,1518,9285,7368,1962,7608,8805,12461,7262,4768,9893,10111,8635,
          12580,2295,9849,4714,7883,6952,12334,76,2268,4205,4934,14062,9815,14787}

def describe(g):
    positive=g.Y>0
    return {'n':len(g),'권역보다덜감소수':int(positive.sum()),'비율':positive.mean(),
            'Y중앙값(%p)':g.Y.median(),'Y평균(%p)':g.Y.mean(),
            'Y25분위':g.Y.quantile(.25),'Y75분위':g.Y.quantile(.75),
            'Y최소':g.Y.min(),'Y최대':g.Y.max(),
            'Y승차량가중평균(%p)':np.average(g.Y,weights=g.board_2024_all),
            '0<Y≤2수':int(((g.Y>0)&(g.Y<=2)).sum()),
            'Y≥20수':int((g.Y>=20).sum()),'Y≥30수':int((g.Y>=30).sum()),
            '기초5미만수':int((g.board_2024_all<5).sum()),
            '기초10미만수':int((g.board_2024_all<10).sum()),
            '기초20미만수':int((g.board_2024_all<20).sum())}

def evaluate():
    original=ranker.load_base_module().load_data()
    assert len(original)==126 and original['셀 ID'].is_unique
    assert original[['nearest_station',*ranker.FEATURES]].notna().all().all()
    assert (original.board_2024_all>0).all() # 현재 자료0명 없음. 임의 +1 보정 없음.
    parts=[];folds=[]
    for station in original.nearest_station.unique():
        train=original[original.nearest_station!=station]
        test=original[original.nearest_station==station].copy()
        c,inner=ranker.choose_c_inner_loso(train)
        x,y=ranker.make_pairs(train);model=ranker.make_model(c).fit(x,y)
        test['ranking_score']=ranker.cell_scores(model,test)
        test['R_s']=100*(test.board_2025_all.sum()/test.board_2024_all.sum()-1)
        test['R_i']=100*(test.board_2025_all/test.board_2024_all-1)
        test['Y']=test.R_i-test.R_s
        assert ((test.Y>0).astype(int)==test.share_increase).all()
        test=test.sort_values('ranking_score',ascending=False)
        test['권역내순위']=np.arange(1,len(test)+1)
        k=int(np.ceil(len(test)*.2));test['상위20%']=test['권역내순위']<=k
        precision=(test.head(k).Y>0).mean()
        folds.append({'검증역':station,'n':len(test),'선택C':c,'내부AUC':inner,
                      '외부AUC':roc_auc_score(test.share_increase,test.ranking_score),
                      '상위수':k,'Precision20':precision,'Lift20':precision/(test.Y>0).mean()})
        parts.append(test)
    oof=pd.concat(parts,ignore_index=True)
    top=oof[oof['상위20%']].copy()
    assert len(top)==27 and int((top.Y>0).sum())==22
    assert set(top['셀 ID'].astype(int))==EXPECTED
    summary=[]
    for station,g in [('전체',oof),*list(oof.groupby('nearest_station',sort=False))]:
        for label,mask in [('상위20%',g['상위20%']),('나머지',~g['상위20%'])]:
            summary.append({'권역':station,'구분':label,**describe(g[mask])})
    diagnostic=[]
    for cutoff in [0,5,10,20]:
        g=top[top.board_2024_all>=cutoff]
        diagnostic.append({'기초기준':cutoff,**describe(g)})
    return oof,top,pd.DataFrame(folds),pd.DataFrame(summary),pd.DataFrame(diagnostic)

def main():
    oof,top,folds,summary,diagnostic=evaluate()
    print('연구목표: 권역 전체 대비 광역버스 수요가 상대적으로 덜 감소할 격자의 권역 내 순위')
    print('기존 4변수:',ranker.FEATURES,'| 기존 LOSO/내부 C 선택 그대로 재현')
    for title,table in [('기존 성능 재현',folds),('실제 상대적 차이',summary),('상위격자의 기초규모별 진단',diagnostic)]:
        print('\n=====',title,'=====');print(table.round(4).to_string(index=False))
    print('\n===== 상위27개 전체 =====')
    print(top[['nearest_station','셀 ID','권역내순위','board_2024_all','board_2025_all','R_s','R_i','Y','ranking_score']].round(4).to_string(index=False))
    print('\nY는 관찰된 차이의 사후검증이며 이 차이의 크기 자체를 학습한 모형이 아니다.')
    print('기초기준은 기존 상위27개를 설명하는 부분집합이며 재학습·순위변경·임계값 선택을 하지 않았다.')
    print('C 실제 성능 입증/변화율 수치 예측이 아니다. 출력 파일 저장 없음.')

if __name__=='__main__':main()
