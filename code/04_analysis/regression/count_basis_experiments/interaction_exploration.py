"""승차건수 기준 126개 격자 OLS 상호작용 탐색. 기존 파일은 유지하며 화면에만 출력.

Main effects are centered (not scaled) before interaction products. All parent
terms remain in the model. No subset selection, clipping, imputation or removal.
LOOCV PRESS and fixed-candidate LOSO are diagnostics, not unbiased estimates
of the performance of a model selected after looking at this comparison.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import linear_reset

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'config.py').is_file() and (p/'data').is_dir())
LABELS = {'walk':'log(1+보행시간)', 'pop':'20~50대 인구', 'floor':'log(1+연면적)',
          'res':'주거지역 비율', 'comp':'경쟁 목적지 비율', 'bus':'버스 통행시간', 'ic':'IC 도로거리'}
PAIRS = [('walk','comp'),('walk','res'),('floor','res'),('walk','floor'),('bus','comp')]
BASE = ['walk','pop','floor','res','comp','bus','ic']

def load():
    d=pd.read_csv(ROOT/'data/final/gtx_a/S1_126격자_건수기준_기초표.csv')
    assert len(d)==126 and d['셀 ID'].is_unique
    X=pd.DataFrame({'walk':np.log1p(d.walk_min), 'pop':d.pop_20_50,
                    'floor':np.log1p(d['2024년 건축물 연면적(천㎡)']),
                    'res':d['주거지역 비율(%)'], 'comp':d.competing_share,
                    'bus':d.bus_time_mean, 'ic':d.ic_access})
    y=d.change.astype(float)
    assert np.isfinite(X.to_numpy()).all() and np.isfinite(y).all()
    np.testing.assert_array_equal(y, d.board_2025-d.board_2024)
    # 교통카드 집계와 무관한 기존 변수만 추가. 이용인원 가중 출근비율 등은 혼용하지 않는다.
    extra_specs = [('26_*.csv','drive_min','drive'),
                   ('08_*.csv','건축물 수','buildings'),
                   ('12_*.csv','2024년 공시지가(원/㎡)','price')]
    for pattern,column,key in extra_specs:
        paths=list((ROOT/'data/processed/gtx_a').glob(pattern))
        assert len(paths)==1,(pattern,paths)
        extra=pd.read_csv(paths[0]).set_index('셀 ID')
        assert extra.index.is_unique
        v=pd.to_numeric(extra[column].reindex(d['셀 ID']),errors='coerce').to_numpy()
        X[key]=np.log1p(v)
    return d,X,y

def design(X,cols,pairs,means):
    z=X[cols]-means[cols]
    for a,b in pairs:z[a+'×'+b]=(X[a]-means[a])*(X[b]-means[b])
    return sm.add_constant(z,has_constant='add')

def specifications():
    for label,cols in [('7변수',BASE),('6변수',[c for c in BASE if c!='ic'])]:
        yield label+' 기본',cols,[]
        for pair in PAIRS:yield label+' + '+pair[0]+'×'+pair[1],cols,[pair]
        for name,pairs in [('보행×경쟁+보행×주거',[PAIRS[0],PAIRS[1]]),
                           ('보행×경쟁+버스×경쟁',[PAIRS[0],PAIRS[4]]),
                           ('보행×주거+연면적×주거',[PAIRS[1],PAIRS[2]])]:
            yield label+' + '+name,cols,pairs
        for key in ['drive','buildings','price']:
            yield label+' + log(1+'+{'drive':'차량시간','buildings':'건축물수','price':'공시지가'}[key]+')',cols+[key],[]
        yield label+' + 로그보행 제곱',cols,[('walk','walk')]

def run():
    d,X,y=load();rows=[];fits={};specs=list(specifications())
    for name,cols,pairs in specs:
        if not np.isfinite(X[cols].to_numpy()).all():
            print('보류(126개 유지 불가, 결측/무한대):',name)
            continue
        z=design(X,cols,pairs,X.mean());plain=sm.OLS(y,z).fit();f=sm.OLS(y,z).fit(cov_type='HC3')
        fits[name]=(f,z,plain)
        h=plain.get_influence().hat_matrix_diag
        loo=float(np.mean(np.abs(plain.resid/(1-h))))
        pred=np.empty(len(y))
        for station in d.nearest_station.unique():
            tr=d.nearest_station!=station;te=~tr;means=X.loc[tr].mean()
            ft=sm.OLS(y[tr],design(X.loc[tr],cols,pairs,means)).fit()
            pred[te]=ft.predict(design(X.loc[te],cols,pairs,means))
        row={'모형':name,'N':int(f.nobs),'R2':f.rsquared,'수정R2':f.rsquared_adj,
             'AIC':f.aic,'최대VIF':max(variance_inflation_factor(z.values,j) for j in range(1,z.shape[1])),
             '최대CookD':float(OLSInfluence(plain).cooks_distance[0].max()),
             'LOOCV_MAE':loo,'LOSO_MAE':np.mean(np.abs(y-pred)),
             'RESET_p':float(linear_reset(plain,power=2,use_f=True).pvalue)}
        for a,b in pairs:
            j=z.columns.get_loc(a+'×'+b)
            row[a+'×'+b+' 계수']=f.params.iloc[j];row[a+'×'+b+' p']=f.pvalues.iloc[j]
        rows.append(row)
    results=pd.DataFrame(rows)
    print('종속변수=2025−2024 승차건수 변화량 / 126개 전체 / HC3 / 평균중심화 / CSV 저장 없음')
    print('상호작용: walk=로그보행, floor=로그연면적, res=주거비율, comp=경쟁비율, bus=버스시간')
    main=['모형','N','R2','수정R2','AIC','최대VIF','최대CookD','LOOCV_MAE','LOSO_MAE','RESET_p']
    print(results[main].round(4).to_string(index=False))
    term_cols=[c for c in results.columns if c.endswith(' p')]
    print('\n상호작용·제곱항 HC3 p값:')
    print(results[['모형']+term_cols].dropna(subset=term_cols,how='all').round(4).to_string(index=False))
    # Fixed reference and all candidates are printed; no automatic final-model adoption.
    for name in [r['모형'] for r in sorted(rows,key=lambda r:r['수정R2'],reverse=True)[:3]]:
        f,z,_=fits[name];ci=f.conf_int()
        table=pd.DataFrame({'변수':z.columns,'계수':f.params.to_numpy(),'HC3 표준오차':f.bse.to_numpy(),'p':f.pvalues.to_numpy(),'CI 하한':ci.iloc[:,0].to_numpy(),'CI 상한':ci.iloc[:,1].to_numpy()})
        print('\n후보:',name);print(table.round(6).to_string(index=False))
    print('\n주의: 전체자료를 보고 비교한 탐색이다. 후보 중 최고 점수는 확정된 일반화 성능이 아니다.')
    print('HC3는 공간상관·누락변수·함수형태 오류를 해결하지 않는다. 상호작용 포함 후 주효과는 다른 변수가 평균일 때의 관계다.')
    return results,fits

if __name__=='__main__':run()
