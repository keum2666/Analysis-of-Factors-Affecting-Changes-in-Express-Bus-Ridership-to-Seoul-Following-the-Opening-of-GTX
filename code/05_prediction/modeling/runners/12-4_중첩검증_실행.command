#!/bin/zsh
# 이 파일 위치 기준으로 이동 (어느 컴퓨터에서도 동작)
cd "$(dirname "$0")/.." || exit 1
# 금경훈 맥북의 tool 가상환경이 있으면 사용, 없으면 python3
PY="$HOME/Desktop/tool/bin/python"; [ -x "$PY" ] || PY="python3"
"$PY" "12-4_변수선택포함_중첩교차검증.py"
printf '\n실행이 끝났습니다. 이 창을 닫아도 됩니다.\n'
read -k 1 '?아무 키나 누르면 창이 닫힙니다.'
