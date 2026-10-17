"""원자료를 제외한 전달용 zip 만들기.

    python make_transfer_zip.py

- data/raw/*의 실제 파일은 제외하고, 재현에 필요한 data/processed/*는 포함한다.
- 한글 파일명을 UTF-8 + NFC로 저장해 윈도우에서도 깨지지 않는다 (맥 기본 zip은 깨짐).
- 결과: 이 폴더의 상위 폴더에 <폴더이름>_전달용.zip
"""
import os
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / f"{ROOT.name}_전달용.zip"
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}
DATA_DIRS = {"raw"}

nfc = lambda text: unicodedata.normalize("NFC", text)

with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        parts = Path(dirpath).relative_to(ROOT).parts
        in_data = len(parts) >= 2 and parts[0] == "data" and parts[1] in DATA_DIRS
        if in_data and len(parts) >= 4:  # data/raw/<자료>/ 까지만 폴더를 남긴다
            continue
        relative = Path(dirpath).relative_to(ROOT.parent)
        archive.writestr(nfc(str(relative)) + "/", "")
        for filename in filenames:
            if filename == ".DS_Store" or in_data:
                continue
            archive.write(Path(dirpath) / filename, nfc(str(relative / filename)))

print(f"저장: {OUTPUT} ({OUTPUT.stat().st_size // 1024:,} KB)")
