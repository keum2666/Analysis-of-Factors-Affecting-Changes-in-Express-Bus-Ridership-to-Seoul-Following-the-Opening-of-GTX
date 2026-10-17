"""18번 상대적 변화율 모형의 사전 고정 0/5/10명 기준 민감도 분석.

권역 R_s는 원래 126개로 고정. 기준을 통과한 표본으로 다시 내부 alpha 선택/외부 LOSO.
같은 외부 검증 격자에서 전체 학습 모형과 제외 후 재학습 모형을 비교한다.
세 임계값에서 최적값을 선택하지 않으므로 임계값 선택 성능이라고 주장하지 않는다.
개통 전 승차량은 대상 선정에만 추가 사용하며 여전히 입력변수에는 넣지 않는다.
결과 CSV/모델/그림 파일은 저장하지 않는다.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('relative_base_19',next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('18_상대적수요변화율_Ridge_LOSO.py')))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
THRESHOLDS=(0,5,10)

def evaluate():
    original,data,totals=base.load_data()
    results={};coverage=[];overall=[];folds=[];ranks=[];tops=[];paired=[]
    for threshold in THRESHOLDS:
        eligible=data[data.board_2024_all>=threshold].copy()
        assert eligible.Y.equals(data.loc[eligible.index,'Y'])
        evaluated=base.evaluate(eligible)
        results[threshold]=evaluated
        for station in base.STATIONS:
            a=original[original.nearest_station==station]
            b=eligible[eligible.nearest_station==station]
            coverage.append({'기준':threshold,'검증역':station,'원래n':len(a),'대상n':len(b),
                             '제외n':len(a)-len(b),'대상비율':len(b)/len(a),
                             '유지된개통전승차비율':b.board_2024_all.sum()/a.board_2024_all.sum(),
                             '기초0명':int((a.board_2024_all==0).sum())})
        for destination,table in zip([overall,folds,ranks,tops],evaluated[1:5]):
            part=table.copy();part.insert(0,'기준',threshold);destination.append(part)
    full=results[0][0].set_index('셀 ID')
    assert abs(results[0][1].loc[lambda x:x['모형']=='Ridge','MAE(%p)'].iloc[0]-20.4013)<.0001
    for threshold in THRESHOLDS:
        fresh=results[threshold][0].set_index('셀 ID')
        same=full.loc[fresh.index]
        assert np.allclose(same.Y,fresh.Y)
        for station in ['전체',*base.STATIONS]:
            mask=np.ones(len(fresh),dtype=bool) if station=='전체' else fresh.nearest_station.eq(station).to_numpy()
            a=same.loc[mask];b=fresh.loc[mask]
            for label,pred in [('전체학습모형의같은대상예측',a.Ridge),('기준적용후재학습',b.Ridge),('Y=0 기준',np.zeros(len(b)))]:
                paired.append({'평가대상기준':threshold,'검증역':station,'모형':label,'n':len(b),
                               **base.metrics(b.Y.to_numpy(),np.asarray(pred))})
    return results,pd.DataFrame(coverage),pd.concat(overall,ignore_index=True),pd.concat(folds,ignore_index=True),pd.concat(ranks,ignore_index=True),pd.concat(tops,ignore_index=True),pd.DataFrame(paired),totals

def target_status(pre_counts,threshold):
    """C 적용 정책의 입력 검사 예시. 적용 제외를 감소·0 예측으로 바꾸지 않는다."""
    s=pd.to_numeric(pd.Series(pre_counts),errors='coerce')
    status=pd.Series('적용 대상',index=s.index)
    status.loc[s<threshold]='기준 미만: 적용 대상 아님'
    status.loc[s==0]='기초 0명: 변화율 정의 불가'
    status.loc[s.isna()|(s<0)]='자료 확인 필요'
    return status

def main():
    results,coverage,overall,folds,ranks,tops,paired,totals=evaluate()
    print('입력4개/원래126개 기준 R_s 고정. 0/5/10명은 사전 고정 민감도이며 최적 기준 선택 안 함.')
    for title,table in [('대상 및 제외',coverage),('전체 OOF',overall),('역별',folds),
                        ('같은 검증격자 비교',paired),('순위',ranks),('상위10/20/30%',tops)]:
        print('\n=====',title,'=====');print(table.round(4).to_string(index=False))
    print('\nGTX-C에 적용하는 경우 기준 미만은 적용 대상 아님/별도 검토로 표시하며 수치 0을 채우지 않는다.')
    print('이번 결과로 임계값을 선택하지 않는다. 향후 선택 시에는 공통 평가대상·적용범위 기준을 정하고 학습자료 내부에서 선택해야 한다.')
    print('CSV/모델/그림 파일 저장 없음.')

if __name__=='__main__':main()
