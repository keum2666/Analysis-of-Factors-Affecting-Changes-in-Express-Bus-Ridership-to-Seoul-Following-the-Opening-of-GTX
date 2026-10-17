"""011-2: 실제/OOF 예측/오차 카카오 지도. 키와 HTML은 메모리에만 보관."""
import errno
import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlparse, parse_qs
import numpy as np
from IPython.display import HTML, display

_server = None
_thread = None
_pages = {}
STATIONS = ['운정중앙', '킨텍스', '대곡']
COORDS = {'운정중앙': [37.716670,126.728330], '킨텍스': [37.665,126.748060], '대곡': [37.631626,126.811024]}
COUNT_COLORS = ['#eff3ff','#bdd7e7','#6baed6','#3182bd','#08519c']
ERROR_COLORS = ['#08519c','#6baed6','#bdd7e7','#ededed','#fcbba1','#fb6a4a','#a50f15']


def index_page():
    """localhost 기본 주소에도 지도 선택 화면을 제공한다."""
    sections=[]
    for station in STATIONS:
        links=[]
        for kind,label in [('actual','실제 승차인원'),('predicted','예측 승차인원'),('error','예측 오차 지도')]:
            url='/ridge?station='+quote(station)+'&kind='+kind
            links.append(f'<a href="{html.escape(url,quote=True)}">{label}</a>')
        sections.append(f'<section><h2>{html.escape(station)} 권역</h2>{" ".join(links)}</section>')
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><title>GTX-A Ridge 지도</title><style>body{font-family:Apple SD Gothic Neo,Malgun Gothic,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#243443}section{padding:14px 0;border-bottom:1px solid #ccc}a{display:inline-block;padding:12px 16px;background:#eef4fa;margin:5px;color:#174c7b;border-radius:6px}p{line-height:1.8}</style><h1>GTX-A 실제 · 예측 · 오차 지도</h1><p>아래 권역과 지도 종류를 선택하세요. 이 화면이 보이면 로컬 지도 서버는 정상 연결된 상태입니다.</p>'+''.join(sections)+'<p>지도 화면에 인증 오류가 나타나면 카카오 JavaScript 키와 현재 주소의 사이트 도메인 등록을 확인하세요. 노트북 커널이 종료되면 이 지도 서버도 종료됩니다.</p></html>'


def stop():
    global _server, _thread
    if _server is not None:
        _server.shutdown()
        _server.server_close()
        if _thread is not None:
            _thread.join(timeout=5)
        _server, _thread = None, None


def build_page(station, kind, cells, model_name, app_key):
    title = {'actual':'실제 2025년 승차인원', 'predicted':'외부검증 예측 승차인원', 'error':'외부검증 오차 (예측 − 실제)'}[kind]
    colors = ERROR_COLORS if kind == 'error' else COUNT_COLORS
    labels = ['−40명 미만','−40~−20명','−20~−5명','±5명 이내','+5~+20명','+20~+40명','+40명 초과'] if kind == 'error' else ['10명 미만','10~40명 미만','40~100명 미만','100~300명 미만','300명 이상']
    legend = ''.join(f'<span><i style="background:{c}"></i>{s}</span>' for c,s in zip(colors,labels))
    payload = json.dumps(cells, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    sdk = 'https://dapi.kakao.com/v2/maps/sdk.js?appkey='+quote(app_key, safe='')+'&autoload=false'
    return '''<!doctype html><html lang="ko"><meta charset="utf-8"><style>
body{margin:0;font-family:Apple SD Gothic Neo,Malgun Gothic,sans-serif;color:#243443}
h3{margin:10px;font-size:17px}#map{height:470px;background:#f2f4f6}.legend{display:flex;flex-wrap:wrap;gap:10px;padding:8px;font-size:12px}
.legend i{display:inline-block;width:13px;height:13px;border:1px solid #777;margin-right:4px;vertical-align:middle}
.note{padding:3px 10px;font-size:12px}.tag{background:white;border:2px solid #222;border-radius:4px;padding:2px 4px;font-weight:bold;font-size:12px;pointer-events:none}
.error{padding:20px;color:#a11;line-height:1.8}
</style><h3>__TITLE__</h3><div id="map"></div><div class="legend">__LEGEND__</div>
<div class="note">__MODEL__ · 모든 권역에 동일한 색상 기준</div>
<div class="note">녹색 테두리 G1~G3: 절대오차 작은 3개 / 노란 테두리 W1~W3: 절대오차 큰 3개</div>
<div class="note"><label><input id="labels" type="checkbox" checked>오차 순위 표시</label> · 격자를 클릭하면 실제값·예측값·좌표를 확인합니다.</div>
<script>
const cells=__CELLS__, kind=__KIND__, stationPoint=__POINT__, overlays=[];
function error(text){document.getElementById('map').innerHTML='<div class="error">'+text+'</div>';}
function initMap(){
 const map=new kakao.maps.Map(document.getElementById('map'),{center:new kakao.maps.LatLng(...stationPoint),level:6});
 const bounds=new kakao.maps.LatLngBounds(), info=new kakao.maps.InfoWindow({removable:true});
 for(const cell of cells){
  for(const rings of cell.paths){
   const path=rings.map(r=>r.map(xy=>{const p=new kakao.maps.LatLng(xy[1],xy[0]);bounds.extend(p);return p;}));
   const highlighted=kind==='error'&&(cell.good||cell.bad);
   const poly=new kakao.maps.Polygon({map,path,strokeColor:highlighted?(cell.bad?'#FFD400':'#009E73'):'#354453',strokeWeight:highlighted?4:1,fillColor:cell.colors[kind],fillOpacity:.72});
   kakao.maps.event.addListener(poly,'click',e=>{
    const node=document.createElement('div');node.style.cssText='padding:10px;line-height:1.7;min-width:230px;white-space:pre-line';
    node.textContent='격자 '+cell.id+'\\n2024년: '+cell.before.toFixed(0)+'명\\n실제 2025년: '+cell.actual.toFixed(0)+'명\\n예측 2025년: '+cell.pred.toFixed(2)+'명\\n오차(예측−실제): '+cell.error.toFixed(2)+'명\\n절대오차: '+Math.abs(cell.error).toFixed(2)+'명\\n보행시간: '+cell.walk.toFixed(1)+'분\\n위도, 경도: '+cell.lat.toFixed(6)+', '+cell.lon.toFixed(6);
    info.setContent(node);info.setPosition(e.latLng);info.open(map);
   });
  }
  if(kind==='error'&&(cell.good||cell.bad)){
   const el=document.createElement('span');el.className='tag';el.textContent=cell.bad?'W'+cell.bad:'G'+cell.good;
   el.style.borderColor=cell.bad?'#b79700':'#009E73';
   const overlay=new kakao.maps.CustomOverlay({map,position:new kakao.maps.LatLng(cell.lat,cell.lon),content:el,yAnchor:.5});overlays.push(overlay);
  }
 }
 new kakao.maps.Marker({map,position:new kakao.maps.LatLng(...stationPoint)});
 map.setBounds(bounds,30,30,30,30);
 document.getElementById('labels').addEventListener('change',e=>overlays.forEach(x=>x.setMap(e.target.checked?map:null)));
}
function loaded(){if(!window.kakao||!kakao.maps){error('지도 SDK 인증 실패. JavaScript 키와 '+location.origin+' 사이트 도메인 등록을 확인하세요.');return;}kakao.maps.load(initMap);}
</script><script src="__SDK__" onload="loaded()" onerror="error('카카오 지도 연결 실패. 네트워크·JavaScript 키·현재 localhost 도메인 등록을 확인하세요.')"></script></html>'''.replace('__TITLE__',html.escape(station+' 권역 · '+title)).replace('__MODEL__',html.escape(model_name)).replace('__LEGEND__',legend).replace('__CELLS__',payload).replace('__KIND__',json.dumps(kind)).replace('__POINT__',json.dumps(COORDS[station])).replace('__SDK__',html.escape(sdk,quote=True))


def make_pages(frame, model_name, app_key):
    assert len(frame)==126 and frame['셀 ID'].nunique()==126
    assert frame[['예측승차인원','board_2025_all','오차','위도','경도','walk_min']].notna().all().all()
    pages={}
    for station in STATIONS:
        cells=[]
        part=frame.to_crs(4326)
        part=part[part.nearest_station==station]
        for _,row in part.iterrows():
            geom=row.geometry.__geo_interface__
            paths=[geom['coordinates']] if geom['type']=='Polygon' else geom['coordinates']
            def count_color(v):
                return COUNT_COLORS[int(np.searchsorted([10,40,100,300],v,side='right'))]
            e=float(row['오차'])
            code=0 if e < -40 else 1 if e < -20 else 2 if e < -5 else 3 if e <= 5 else 4 if e <= 20 else 5 if e <=40 else 6
            cells.append({'id':int(row['셀 ID']),'paths':paths,'lat':float(row['위도']),'lon':float(row['경도']),
                          'before':float(row.board_2024_all),'actual':float(row.board_2025_all),
                          'pred':float(row['예측승차인원']),'error':e,'walk':float(row.walk_min),
                          'good':int(row['작은오차순위']) if row['작은오차순위']<=3 else 0,
                          'bad':int(row['큰오차순위']) if row['큰오차순위']<=3 else 0,
                          'colors':{'actual':count_color(row.board_2025_all),'predicted':count_color(row['예측승차인원']),'error':ERROR_COLORS[code]}})
        for kind in ['actual','predicted','error']:
            pages[(station,kind)]=build_page(station,kind,cells,model_name,app_key)
    return pages


def show_maps(frame, model_name, app_key):
    global _server,_thread,_pages
    if not app_key:
        raise ValueError('카카오 JavaScript 키를 입력하세요.')
    _pages=make_pages(frame,model_name,app_key)
    # 같은 모듈의 서버를 재사용하므로 셀 재실행 시 포트 충돌이 없다.
    if _server is None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed=urlparse(self.path); params=parse_qs(parsed.query)
                key=(params.get('station',[''])[0],params.get('kind',[''])[0])
                if parsed.path in ('/', '/index.html'):
                    body=index_page().encode('utf-8')
                elif parsed.path=='/ridge' and key in _pages:
                    body=_pages[key].encode('utf-8')
                else:
                    self.send_error(404);return
                self.send_response(200)
                self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Cache-Control','no-store')
                self.send_header('Content-Length',str(len(body)))
                self.end_headers();self.wfile.write(body)
            def log_message(self,*args):
                pass
        for port in [8765,8766,8767,8768]:
            try:
                _server=ThreadingHTTPServer(('127.0.0.1',port),Handler);break
            except OSError as exc:
                if exc.errno!=errno.EADDRINUSE: raise
        if _server is None:
            raise RuntimeError('8765~8768 포트가 사용 중입니다. 이전 지도 노트북 커널을 종료하세요.')
        _thread=threading.Thread(target=_server.serve_forever,daemon=True);_thread.start()
    port=_server.server_port
    print(f'지도 시작 주소 / 카카오 등록 도메인: http://localhost:{port}')
    display(HTML(f'<p><a href="http://localhost:{port}/" target="_blank">권역별 지도 선택 화면 열기</a></p>'))
    for station in STATIONS:
        def iframe(kind):
            url=f'http://localhost:{port}/ridge?station={quote(station)}&kind={kind}'
            return f'<iframe title="{html.escape(station+kind)}" src="{html.escape(url,quote=True)}" style="width:100%;height:625px;border:1px solid #b8c4ce;border-radius:6px"></iframe><a href="{html.escape(url,quote=True)}" target="_blank">새 창으로 크게 보기</a>'
        display(HTML(f'<h3>{html.escape(station)} 권역 · 실제 / 예측 / 오차</h3><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px"><div>{iframe("actual")}</div><div>{iframe("predicted")}</div></div><div style="margin:14px 0 30px">{iframe("error")}</div>'))
