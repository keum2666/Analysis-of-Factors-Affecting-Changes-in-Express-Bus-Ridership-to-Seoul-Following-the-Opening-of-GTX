"""개통 전 X만으로 고정한 4유형의 실제 수요 변화 비교. C 미적용.

보행시간은 전체126 중앙값, 경쟁비율은 0과 양수로 구분한다.
원래 126개 기준 R_s와 유형 기준은 기초승차량 부분집합에서도 바꾸지 않는다.
통계적 유의성/인과효과/새 지역 예측으로 해석하지 않는다. 출력 파일 저장 없음.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('type_base',next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('18_상대적수요변화율_Ridge_LOSO.py')))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
ORDER=['A','B','C','D']
NAMES={'A':'상대짧음·경쟁양수','B':'상대짧음·경쟁0','C':'상대긺·경쟁양수','D':'상대긺·경쟁0'}

def summarize(g):
    w=g.board_2024_all.to_numpy();total=w.sum()
    return {'n':len(g),'기초5미만':int((g.board_2024_all<5).sum()),
            '기초10미만':int((g.board_2024_all<10).sum()),
            '기초20미만':int((g.board_2024_all<20).sum()),
            '승차전':g.board_2024_all.sum(),'승차후':g.board_2025_all.sum(),
            '변화량합':(g.board_2025_all-g.board_2024_all).sum(),
            '기초승차중앙값':g.board_2024_all.median(),
            '변화율평균(%)':g.R_i.mean(),'변화율중앙값(%)':g.R_i.median(),
            '총량변화율(%)':100*(g.board_2025_all.sum()/total-1) if total>0 else np.nan,
            'Y평균(%p)':g.Y.mean(),'Y중앙값(%p)':g.Y.median(),
            'Y가중평균(%p)':np.average(g.Y,weights=w) if total>0 else np.nan,
            'Y_IQR(%p)':g.Y.quantile(.75)-g.Y.quantile(.25),
            'Y>0비율':(g.Y>0).mean(),'소표본주의(n<5)':len(g)<5}

def analyze():
    original,data,totals=base.load_data()
    if original[['walk_min','competing_share']].isna().any().any():
        raise ValueError('유형 변수 결측 확인 필요: 결측을 0으로 대체하지 않는다.')
    cutoff=float(original.walk_min.median())
    data['유형']=np.select([(data.walk_min<=cutoff)&(data.competing_share>0),
                           (data.walk_min<=cutoff)&(data.competing_share==0),
                           (data.walk_min>cutoff)&(data.competing_share>0)],['A','B','C'],default='D')
    assert len(data)==126 and data.유형.notna().all()
    definition=pd.DataFrame([{'유형':t,'설명':NAMES[t],
                     '보행기준':f'≤ {cutoff:.3f}분' if t in ['A','B'] else f'> {cutoff:.3f}분',
                     '경쟁기준':'> 0%' if t in ['A','C'] else '= 0%'} for t in ORDER])
    counts=pd.crosstab(data.nearest_station,data.유형).reindex(index=base.STATIONS,columns=ORDER,fill_value=0)
    tables=[];contrasts=[]
    # 결과를 보고 기준을 고르지 않는다. 5/10명은 동일 유형의 고정 민감도 기술통계다.
    for threshold in [0,5,10]:
        sub=data[data.board_2024_all>=threshold]
        for station,g in [('전체',sub),*list(sub.groupby('nearest_station',sort=False))]:
            summaries={}
            for t,part in g.groupby('유형',sort=True):
                row=summarize(part);summaries[t]=row
                tables.append({'기초기준':threshold,'권역':station,'유형':t,**row})
            # 같은 기초기준 안에서 방향이 역마다 반복되는지 보는 기술적 대비.
            for label,l,r in [('경쟁양수에서_짧음-긺','A','C'),('경쟁0에서_짧음-긺','B','D'),
                              ('보행짧음에서_경쟁양수-0','A','B'),('보행긺에서_경쟁양수-0','C','D')]:
                if l in summaries and r in summaries:
                    a,b=summaries[l],summaries[r]
                    contrasts.append({'기초기준':threshold,'권역':station,'대비':label,
                                     '왼쪽n':a['n'],'오른쪽n':b['n'],
                                     'Y중앙값차이(%p)':a['Y중앙값(%p)']-b['Y중앙값(%p)'],
                                     'Y가중평균차이(%p)':a['Y가중평균(%p)']-b['Y가중평균(%p)']})
    # 기존 20분 기준을 고집할 때의 표본 분포도 함께 보고하되 결과에 맞춰 선택하지 않는다.
    twenty=pd.crosstab(data.nearest_station,np.select([(data.walk_min<=20)&(data.competing_share>0),
                           (data.walk_min<=20)&(data.competing_share==0),
                           (data.walk_min>20)&(data.competing_share>0)],['A','B','C'],default='D')).reindex(index=base.STATIONS,columns=ORDER,fill_value=0)
    return data,definition,counts,pd.DataFrame(tables),pd.DataFrame(contrasts),twenty,totals

def main():
    data,definition,counts,tables,contrasts,twenty,totals=analyze()
    print('유형 기준은 X만 사용. 보행 전체 중앙값:',float(data.walk_min.median()),'분; 경쟁0:',int((data.competing_share==0).sum()))
    for title,table in [('유형 정의',definition),('유형별 표본',counts),('전체표본 유형별 결과',tables[tables.기초기준==0]),
                        ('역별 대비',contrasts[contrasts.기초기준==0]),('10명 이상 기술통계',tables[tables.기초기준==10]),
                        ('기존20분 기준의 표본 분포',twenty)]:
        print('\n=====',title,'=====');print(table.round(3).to_string())
    print('\n경쟁0은 현재 구축값이 0임을 뜻하며 실제 경쟁 부재를 확정하지 않는다. 원자료 좌표/분류가능 수요 확인이 필요하다.')
    print('단순평균은 소규모 격자 영향, 총량/가중평균은 큰 수요 거점 영향을 받는다. 둘 다 확인한다.')
    print('C 미적용. OLS 수정 없음. 결과/그림 파일 저장 없음.')

if __name__=='__main__':main()
