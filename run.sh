#!/bin/sh
# Полный прогон: события T-ECD → таблицы пользователей → A/A-тесты, мощность, подглядывание → reports/.
# Данные скачиваются отдельно: sh data/download.sh
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
.venv/bin/python src/build.py
PYTHONPATH=src .venv/bin/python src/analysis.py > /dev/null
echo "Готово: reports/results.json, reports/figures"
