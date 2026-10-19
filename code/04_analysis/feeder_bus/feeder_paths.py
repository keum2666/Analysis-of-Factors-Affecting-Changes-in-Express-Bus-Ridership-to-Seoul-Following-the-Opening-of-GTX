"""전달 자료의 경로를 UROP 상대경로로 연결. API나 분석을 실행하지 않는다."""
from pathlib import Path
import json, os, unicodedata
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'config.py').is_file() and (p/'data').is_dir())
_TABLE = json.loads(Path(__file__).with_name('transfer_paths.json').read_text(encoding='utf-8'))
def transfer_path(value):
    s = unicodedata.normalize('NFC', str(value)).replace('\\','/')
    if s in _TABLE['files']:
        rel = _TABLE['files'][s]
    else:
        rel = None
        for old, new in sorted(_TABLE['prefixes'].items(), key=lambda x: -len(x[0])):
            if s == old or s.startswith(old+'/'):
                rel = new+s[len(old):]
                break
        if rel is None:
            raise FileNotFoundError('전달 경로표에 없는 경로: '+s)
    p = ROOT/rel
    # 재현 검증은 결과만 임시 폴더로 보내며 원본 산출물은 덮어쓰지 않는다.
    temp = os.environ.get('UROP_FEEDER_TEST_RESULTS')
    if temp and rel.startswith('results/analysis/feeder_bus/'):
        p = Path(temp)/rel[len('results/analysis/feeder_bus/'):]
    return str(p)
def station_path(argv):
    return argv[1] if len(argv)>1 else str(ROOT/'data/raw/common/smartcard_20241017/STTN_20241017.parquet')

def korean_font():
    from matplotlib import font_manager
    candidates = [Path('/System/Library/Fonts/AppleSDGothicNeo.ttc'),
                  Path('/Library/Fonts/Arial Unicode.ttf'), Path(r'C:\Windows\Fonts\malgun.ttf')]
    for p in candidates:
        if p.is_file():
            font_manager.fontManager.addfont(str(p))
            return font_manager.FontProperties(fname=str(p)).get_name()
    return 'sans-serif'
