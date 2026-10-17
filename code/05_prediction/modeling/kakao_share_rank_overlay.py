"""011-1: 카카오맵에서 GTX-A 실제 비중 확대와 외부검증 순위를 표시한다."""

from __future__ import annotations

import html
import json
import errno
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
from IPython.display import HTML, display


PORT = 8765  # 기존 011에서 등록한 카카오 JavaScript 도메인을 재사용
STATIONS = ("운정중앙", "킨텍스", "대곡")
_server = None
_thread = None


def _paths(geometry):
    shape = geometry.__geo_interface__
    if shape["type"] == "Polygon":
        return [shape["coordinates"]]
    if shape["type"] == "MultiPolygon":
        return shape["coordinates"]
    raise ValueError(f"다각형 격자가 아닙니다: {shape['type']}")


def _page(station, kind, features, station_point, app_key):
    payload = json.dumps(features, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    point = json.dumps(station_point)
    sdk = "https://dapi.kakao.com/v2/maps/sdk.js?appkey=" + quote(app_key, safe="") + "&autoload=false"
    observed = kind == "actual"
    title = "실제 역 권역 내 수요 비중 확대 여부" if observed else "중첩 LOSO 외부검증 예측순위"
    legend = (
        '<span><i style="background:#0E8F7C"></i>실제 비중 확대</span>'
        '<span><i style="background:#BFC7CA"></i>실제 비확대</span>'
        '<span><i style="background:#FFD400"></i>노란 테두리·숫자: 실제 비중 증가폭 TOP 3</span>'
        if observed else
        '<span><i style="background:#165DAB"></i>예측순위 상위 20%(참고 강조)</span>'
        '<span><i style="background:#A8CCE5"></i>그 외 격자</span>'
        '<span>□ 숫자: 권역 내 예측순위</span>'
    )
    note = (
        '비중 = 해당 격자 승차인원 ÷ 같은 역 권역 총승차인원. TOP 3은 비중 증가폭(%p) 기준입니다.'
        if observed else
        '모든 격자에 외부검증 예측순위가 있습니다. 상위 20%는 참고 표시이며, 확대 확률이나 0/1 판정이 아닙니다.'
    )
    toggle = '' if observed else '<label><input type="checkbox" id="show-all"> 모든 격자의 순위 숫자 보기</label>'
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body {{font-family:Apple SD Gothic Neo,Malgun Gothic,sans-serif;margin:0;color:#1d2730}}
.head {{padding:10px 14px;font-size:17px;font-weight:700}}
#map {{height:505px;width:100%;background:#eee}}
.legend {{display:flex;flex-wrap:wrap;gap:12px;padding:8px 14px;font-size:13px;align-items:center}}
.legend span {{display:inline-flex;gap:5px;align-items:center}}
.legend i {{display:inline-block;width:16px;height:16px;border:1px solid #555}}
.note {{padding:0 14px 10px;font-size:12px;color:#45515b}}
.rank {{display:inline-block;background:white;border:1.7px solid #153957;border-radius:3px;
 min-width:20px;padding:1px 3px;text-align:center;font-size:11px;font-weight:800;
 color:#102b44;box-shadow:0 1px 3px #0007;pointer-events:none}}
.rank.top {{background:#FFE46A;border-color:#926600}}
.error {{padding:20px;color:#9b1c1c}}
</style></head><body>
<div class="head">{html.escape(station)} 권역 · {title}</div><div id="map"></div>
<div class="legend">{legend}{toggle}</div><div class="note">{note} 격자를 클릭하면 상세값이 나옵니다.</div>
<script>
const cells={payload}, stationPoint={point}, kind={json.dumps(kind)};
function error(msg) {{document.getElementById('map').innerHTML='<div class="error">'+msg+'</div>';}}
function initMap() {{
 if (!window.kakao || !kakao.maps) {{error('카카오 지도 SDK를 불러오지 못했습니다. 키와 현재 localhost 지도 주소의 도메인 등록을 확인하세요.');return;}}
 const map=new kakao.maps.Map(document.getElementById('map'),{{center:new kakao.maps.LatLng(stationPoint[0],stationPoint[1]),level:6}});
 const bounds=new kakao.maps.LatLngBounds(), info=new kakao.maps.InfoWindow({{removable:true}}), overlays=[];
 for (const cell of cells) {{
  for (const rings of cell.paths) {{
   const path=rings.map(r=>r.map(xy=>{{const p=new kakao.maps.LatLng(xy[1],xy[0]);bounds.extend(p);return p;}}));
   const poly=new kakao.maps.Polygon({{map,path,strokeWeight:cell.top3?4:1.1,
    strokeColor:cell.top3?'#FFD400':'#34424E',strokeOpacity:1,fillColor:cell.color,fillOpacity:.78}});
   kakao.maps.event.addListener(poly,'click',e=>{{
    const detail=kind==='actual'
      ? `승차인원: ${{cell.before}}명 → ${{cell.after}}명<br>권역 내 비중: ${{cell.pre.toFixed(2)}}% → ${{cell.post.toFixed(2)}}%<br>비중 변화: ${{cell.delta>=0?'+':''}}${{cell.delta.toFixed(2)}}%p<br>실제 비중 확대: ${{cell.expanded?'예':'아니오'}}${{cell.top3?'<br>비중 증가폭 TOP '+cell.top3:''}}`
      : `권역 내 예측순위: ${{cell.rank}}위 / ${{cells.length}}개<br>실제 비중 확대: ${{cell.expanded?'예':'아니오'}}`;
    info.setContent(`<div style="padding:8px 11px;line-height:1.55;white-space:nowrap"><b>격자 ${{cell.id}}</b><br>${{detail}}</div>`);
    info.setPosition(e.latLng);info.open(map);
   }});
  }}
  const label=kind==='actual'?cell.top3:cell.rank;
  if(label) {{
   const overlay=new kakao.maps.CustomOverlay({{
    position:new kakao.maps.LatLng(cell.lat,cell.lon),
    content:`<span class="rank ${{cell.top3?'top':''}}">${{label}}</span>`,yAnchor:.5
   }});
   overlays.push({{overlay,initial:kind==='actual'||cell.top20}});
  }}
 }}
 const showAll=document.getElementById('show-all');
 function sync() {{for(const x of overlays) x.overlay.setMap(x.initial||showAll?.checked?map:null);}}
 if(showAll) showAll.addEventListener('change',sync);sync();
 const marker=new kakao.maps.Marker({{map,position:new kakao.maps.LatLng(stationPoint[0],stationPoint[1])}});
 kakao.maps.event.addListener(marker,'click',()=>{{info.setContent('<div style="padding:8px 10px">GTX-A {html.escape(station)}역</div>');info.setPosition(marker.getPosition());info.open(map);}});
 if(cells.length)map.setBounds(bounds,40,40,40,40);
}}
</script><script src="{sdk}" onload="kakao.maps.load(initMap)" onerror="error('카카오 지도 SDK 연결 실패')"></script>
</body></html>"""


def show_share_rank_maps(frame, station_coords, app_key):
    """126개 격자의 실제 3장과 OOF 예측순위 3장을 화면에만 표시한다."""
    global _server, _thread
    required = {"셀 ID", "nearest_station", "board_2024_all", "board_2025_all", "share_pre", "share_post", "share_increase", "ranking_score", "geometry"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"지도 자료에 열이 없습니다: {sorted(missing)}")
    if len(frame) != 126 or frame["셀 ID"].nunique() != 126:
        raise ValueError("고유한 GTX-A 분석 격자 126개가 필요합니다.")
    if not app_key:
        raise ValueError("카카오 JavaScript 키가 필요합니다.")
    if _server is not None:
        _server.shutdown()
        _server.server_close()
        _thread.join(timeout=5)
        _server = None
    # 같은 노트북 커널에서 011 지도를 먼저 실행했다면 이전 서버를 자동 종료한다.
    previous = sys.modules.get("kakao_grid_overlay")
    if previous is not None and getattr(previous, "_server", None) is not None:
        previous._server.shutdown()
        previous._server.server_close()
        if getattr(previous, "_thread", None) is not None:
            previous._thread.join(timeout=5)
        previous._server = None

    geo = frame.to_crs("EPSG:4326")
    pages = {}
    for station in STATIONS:
        part = geo[geo["nearest_station"].astype(str) == station].copy()
        if part.empty:
            raise ValueError(f"{station} 권역 자료가 없습니다.")
        part["비중 변화(%p)"] = (part["share_post"] - part["share_pre"]) * 100
        expanded = part[part["share_increase"] == 1]
        top3 = {int(cell): rank for rank, cell in enumerate(
            expanded.sort_values(["비중 변화(%p)", "셀 ID"], ascending=[False, True])["셀 ID"].head(3), 1
        )}
        part = part.sort_values(["ranking_score", "셀 ID"], ascending=[False, True])
        rank_map = {int(cell): rank for rank, cell in enumerate(part["셀 ID"], 1)}
        top_k = int(np.ceil(len(part) * .20))
        actual_cells, predicted_cells = [], []
        for _, row in part.iterrows():
            cell_id = int(row["셀 ID"])
            center = row.geometry.representative_point()
            base = {
                "id": cell_id, "paths": _paths(row.geometry),
                "lat": center.y, "lon": center.x,
                "expanded": bool(row["share_increase"]),
            }
            actual_cells.append({**base,
                "color": "#0E8F7C" if base["expanded"] else "#BFC7CA",
                "top3": top3.get(cell_id, 0), "pre": float(row["share_pre"]) * 100,
                "post": float(row["share_post"]) * 100,
                "before": int(row["board_2024_all"]), "after": int(row["board_2025_all"]),
                "delta": float(row["비중 변화(%p)"]),
            })
            predicted_cells.append({**base,
                "color": "#165DAB" if rank_map[cell_id] <= top_k else "#A8CCE5",
                "top3": 0, "top20": rank_map[cell_id] <= top_k,
                "rank": rank_map[cell_id],
            })
        pages[(station, "actual")] = _page(station, "actual", actual_cells, station_coords[station], app_key)
        pages[(station, "predicted")] = _page(station, "predicted", predicted_cells, station_coords[station], app_key)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            key = (params.get("station", [""])[0], params.get("kind", [""])[0])
            if parsed.path != "/map" or key not in pages:
                self.send_error(404)
                return
            body = pages[key].encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    selected_port = None
    for candidate_port in (PORT, PORT + 1):
        try:
            _server = ThreadingHTTPServer(("127.0.0.1", candidate_port), Handler)
            selected_port = candidate_port
            break
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise RuntimeError(f"로컬 지도 서버를 열 수 없습니다: {exc}") from exc
            if candidate_port == PORT + 1:
                raise RuntimeError("localhost:8765·8766 포트를 모두 사용 중입니다. 이전 지도 노트북의 커널을 종료하세요.") from exc
    if selected_port != PORT:
        print("8765 포트를 다른 커널이 사용 중이어서 8766으로 실행합니다.")
        print("지도가 빈 화면이면 카카오 JavaScript 키의 사이트 도메인에 http://localhost:8766 을 추가하세요.")
    _thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _thread.start()
    for station in STATIONS:
        actual_url = f"http://localhost:{selected_port}/map?station={quote(station)}&kind=actual"
        predicted_url = f"http://localhost:{selected_port}/map?station={quote(station)}&kind=predicted"
        display(HTML(f"""
        <div style="font-family:sans-serif;margin:18px 0 28px">
          <div style="font-size:19px;font-weight:700;margin:0 0 8px">{html.escape(station)} 권역 · 실제 / 예측</div>
          <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start">
            <iframe title="{html.escape(station)} 실제 비중 확대" src="{html.escape(actual_url, quote=True)}"
                    style="flex:1 1 calc(50% - 8px);min-width:330px;height:625px;border:1px solid #bcc5cc;border-radius:5px"></iframe>
            <iframe title="{html.escape(station)} 외부검증 예측순위" src="{html.escape(predicted_url, quote=True)}"
                    style="flex:1 1 calc(50% - 8px);min-width:330px;height:625px;border:1px solid #bcc5cc;border-radius:5px"></iframe>
          </div>
        </div>"""))
