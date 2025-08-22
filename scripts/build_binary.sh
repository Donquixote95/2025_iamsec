#!/usr/bin/env bash
set -e
APP=iamsec-dtms
rm -rf build dist
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
pyinstaller -F -n ${APP} -p . dtms/cli.py
echo "Done. ./dist/${APP} 생성"