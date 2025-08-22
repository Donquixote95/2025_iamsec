# 2025_iamsec

2025_iamsec/
├─ dtms/                     # 라이브러리(핵심 로직)
│  ├─ __init__.py
│  ├─ converter.py           # 기존 dtms_to_csv.py 로직 이관 (함수화)
│  └─ cli.py                 # CLI 엔트리 포인트 (콘솔용)
├─ app/
│  └─ streamlit_app.py       # 웹 UI (Streamlit)
├─ scripts/
│  ├─ build_binary.sh        # PyInstaller 빌드 스크립트(맥/리눅스)
│  └─ run_app.sh             # Streamlit 실행 스크립트
├─ pyproject.toml            # 패키징/설치(콘솔 명령 등록)
├─ requirements.txt
├─ .gitignore
└─ README.md
