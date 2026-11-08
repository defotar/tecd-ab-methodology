#!/bin/sh
# Скачивает события T-ECD (small) и докачивает то, что не скачалось. Запуск из корня проекта: sh data/download.sh
R=data/raw
BASE=https://huggingface.co/datasets/t-tech/T-ECD/resolve/main/dataset/small
export R BASE
mkdir -p $R/marketplace/events $R/retail/events
for f in users.pq brands.pq; do [ -f "$R/$f" ] || curl -sfL --retry 5 -o "$R/$f" "$BASE/$f"; done
xargs -P 8 -L 1 sh -c 'f=$0; s=$1; out=$R/$f; if [ -f "$out" ] && [ "$(stat -f%z "$out")" = "$s" ]; then exit 0; fi; curl -sfL --retry 5 --retry-delay 5 --max-time 900 -C - -o "$out" "$BASE/$f" || echo "FAIL $f"' < data/files.txt
n=0; while read -r f s; do [ -f "$R/$f" ] && [ "$(stat -f%z "$R/$f")" = "$s" ] || n=$((n+1)); done < data/files.txt
echo "Не скачано файлов: $n"
