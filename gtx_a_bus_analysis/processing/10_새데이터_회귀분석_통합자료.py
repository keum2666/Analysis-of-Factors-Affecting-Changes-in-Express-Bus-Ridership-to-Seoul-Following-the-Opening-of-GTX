"""2024-10-17→2025-11-06, 새 126개 격자의 회귀분석 통합자료를 구축한다."""
from pathlib import Path
import re
import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
P = PROJECT / "data" / "processed"
STABLE = P / "16_새데이터_공급구조유지_250m격자.csv"
WALK = P / "17_새데이터_GTX역_보행접근성_250m격자.csv"
OUT = P / "18_새데이터_회귀분석_격자통합데이터.csv"
EXTRA = P / "18-1_새데이터_추가피처_250m격자.csv"
TIME_OUT = P / "18-2_새데이터_시간대별_승차인원_셀.csv"
CARD = ROOT / "gtx_a_seoul_bus_outputs" / "transport_card"
F24 = CARD / "gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv"
F25 = CARD / "gtx_a_transport_card_20251106_raw_with_coords.csv"

def read(path): return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
def ids(text): return [x.strip() for x in str(text).split("|") if x.strip() and x.strip().lower() != "nan"]
def hav(lat1, lon1, lat2, lon2):
    r=6371.0088; p1,p2=np.radians(lat1),np.radians(lat2); dp=np.radians(lat2-lat1); dl=np.radians(lon2-lon1)
    return 2*r*np.arcsin(np.sqrt(np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2))
def weighted_median(v,w):
    o=np.argsort(v); v=np.asarray(v)[o]; w=np.asarray(w)[o]; return float(v[np.searchsorted(np.cumsum(w),w.sum()/2)])

stable=read(STABLE); stable["셀 ID"]=pd.to_numeric(stable["셀 ID"],errors="coerce").astype(int)
walk=read(WALK); walk["셀 ID"]=pd.to_numeric(walk["셀 ID"],errors="coerce").astype(int)
if len(stable)!=126 or len(walk)!=126: raise ValueError(f"stable={len(stable)}, walk={len(walk)}; 126개가 필요합니다.")
points=gpd.GeoDataFrame(stable[["셀 ID"]].copy(),geometry=gpd.points_from_xy(stable["중심점 경도"],stable["중심점 위도"]),crs="EPSG:4326").to_crs("EPSG:5179")

def join_layers(paths, outcol, agg="sum", value_col=None):
    layers=[]
    for path in paths:
        g=gpd.read_file(path)
        source_col = value_col or next((c for c in ("val", "value") if c in g.columns), None)
        if source_col is None:
            raise KeyError(f"No value column in {path}; columns={g.columns.tolist()}")
        g=g[[source_col,"geometry"]].to_crs("EPSG:5179"); g[outcol]=pd.to_numeric(g[source_col],errors="coerce"); layers.append(g[[outcol,"geometry"]])
    allg=gpd.GeoDataFrame(pd.concat(layers,ignore_index=True),crs="EPSG:5179")
    j=gpd.sjoin(points,allg,how="left",predicate="within")
    return getattr(j.groupby("셀 ID")[outcol],agg)().reindex(stable["셀 ID"]).reset_index()

sgis=PROJECT/"data"/"raw"/"sgis_2024"
pop_parts=[]
for age in ["20대","30대","40대","50대"]:
    paths=list(sgis.rglob(f"*{age}*202410/vl_blk.shp")); pop_parts.append(join_layers(paths,f"pop_{age}","sum"))
pop=pop_parts[0]
for x in pop_parts[1:]: pop=pop.merge(x,on="셀 ID",how="outer")
pop["pop_20_50"]=pop[[f"pop_{a}" for a in ["20대","30대","40대","50대"]]].sum(axis=1,min_count=1)

building_paths=list((ROOT/"Analisis3"/"DATA"/"건축물수").rglob("*202410/vl_blk.shp"))
build=join_layers(building_paths,"buildings","sum")
ic_path=ROOT/"Analisis2"/"output"/"IC"/"(B100)국토통계_국토정책지표-IC 접근성-250M_2024"/"76.1 IC(읍면동격자) 접근성.shp"
ic=join_layers([ic_path],"ic_access","mean",value_col="value")
land_roots=[PROJECT/"data"/"raw"/"land_price",ROOT/"Analisis2"/"output"/"토지_공시지가"]
land_paths=[]
for r in land_roots:
    if r.exists(): land_paths=list(r.glob("*202407/vl_blk.shp")); break
land=join_layers(land_paths,"land_price_2024","mean"); land["log_land_price_2024"]=np.where(land["land_price_2024"]>0,np.log(land["land_price_2024"]),np.nan)

maps={}
for year,col in [(2024,"2024년 정류장 ID"),(2025,"2025년 정류장 ID")]:
    m={}
    for _,r in stable.iterrows():
        for sid in ids(r[col]): m[sid]=int(r["셀 ID"])
    maps[year]=m

def card_frame(path,year):
    d=read(path); d["sid"]=d["ride_sttn_id"].astype(str).str.replace(r"\.0$","",regex=True); d["셀 ID"]=d["sid"].map(maps[year]); d["users"]=pd.to_numeric(d["utztn_nope"],errors="coerce").fillna(0); d["hour"]=pd.to_datetime(d["ride_dt"].astype(str).str.replace(r"\.0$","",regex=True),format="%Y%m%d%H%M%S",errors="coerce").dt.hour
    return d[d["셀 ID"].notna()].copy()
d24=card_frame(F24,2024); d25=card_frame(F25,2025)

bands={"새벽":(0,5),"출근":(6,9),"낮":(10,15),"저녁":(16,23)}; time_rows=[]
for label,(lo,hi) in bands.items():
    a=d24[d24["hour"].between(lo,hi)].groupby("셀 ID")["users"].sum(); b=d25[d25["hour"].between(lo,hi)].groupby("셀 ID")["users"].sum()
    for cid in stable["셀 ID"]: time_rows.append({"셀 ID":cid,"시간대":label,"board_2024":a.get(cid,0),"board_2025":b.get(cid,0),"change":b.get(cid,0)-a.get(cid,0)})
pd.DataFrame(time_rows).to_csv(TIME_OUT,index=False,encoding="utf-8-sig")

# 2024 셀별 경쟁 목적지 비율
alat=pd.to_numeric(d24["하차위도"],errors="coerce"); alon=pd.to_numeric(d24["하차경도"],errors="coerce")
valid=alat.notna()&alon.notna(); dist1=hav(alat,alon,37.618855,126.920859); dist2=hav(alat,alon,37.555980,126.972091)
d24["classified_users"]=np.where(valid,d24["users"],0); d24["competitive_users"]=np.where(valid&((dist1<=1)|(dist2<=1)),d24["users"],0)
comp=d24.groupby("셀 ID")[["classified_users","competitive_users"]].sum(); comp["competing_share"]=np.where(comp["classified_users"]>0,comp["competitive_users"]/comp["classified_users"]*100,0); comp=comp[["competing_share"]].reset_index()

# 2024 셀별 광역버스 통행시간
ride=pd.to_datetime(d24["ride_dt"].astype(str).str.replace(r"\.0$","",regex=True),format="%Y%m%d%H%M%S",errors="coerce"); goff=pd.to_datetime(d24["goff_dt"].astype(str).str.replace(r"\.0$","",regex=True),format="%Y%m%d%H%M%S",errors="coerce"); mins=(goff-ride).dt.total_seconds()/60; mins=np.where(mins<0,mins+1440,mins); d24["travel_min"]=mins
v=d24[(d24["travel_min"]>0)&(d24["travel_min"]<=240)&(d24["users"]>0)]
travel=[]
for cid,g in v.groupby("셀 ID"):
    travel.append({"셀 ID":cid,"bus_time_mean":np.average(g["travel_min"],weights=g["users"]),"bus_time_median":weighted_median(g["travel_min"].to_numpy(),g["users"].to_numpy())})
travel=pd.DataFrame(travel)

model=stable[["셀 ID","중심점 위도","중심점 경도","2024년 승차인원","2025년 승차인원","승차인원 증감"]].rename(columns={"2024년 승차인원":"board_2024_all","2025년 승차인원":"board_2025_all","승차인원 증감":"change_all"})
w=walk[["셀 ID","ORS GET 보행시간(분)","최근접 GTX-A역"]].rename(columns={"ORS GET 보행시간(분)":"walk_min","최근접 GTX-A역":"nearest_station"})
for x in [w,ic,pop[["셀 ID","pop_20_50"]],build,comp]: model=model.merge(x,on="셀 ID",how="left",validate="one_to_one")
model["walk_20"]=(model["walk_min"]<=20).astype(float); model["walk_20_60"]=((model["walk_min"]>20)&(model["walk_min"]<=60)).astype(float)
model.to_csv(OUT,index=False,encoding="utf-8-sig")
extra=stable[["셀 ID"]].merge(land,on="셀 ID",how="left").merge(travel,on="셀 ID",how="left"); extra.to_csv(EXTRA,index=False,encoding="utf-8-sig")

required=["change_all","walk_min","ic_access","competing_share","pop_20_50","buildings"]
print("SAVE",OUT); print("ROWS",len(model)); print("MISSING",model[required].isna().sum().to_dict()); print(model[required].describe().T)
if len(model)!=126 or model[required].isna().any().any(): raise RuntimeError("새 통합자료에 결측이 있습니다.")
