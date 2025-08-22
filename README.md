# 2025 IMASEC – DTMS PDF → CSV Converter

**DTMS-like PDF 문서**를 입력받아, ** **

여단/대대별 활동 계획을 **CSV 파일**로 변환하는 툴입니다. ** **

**-** **주(week) 헤더 기반 파싱**

**-** **세로 병합셀 처리** (공통/단일 라벨 자동 판정, IoU 기반)

**-** **CLI / Web UI / 단일 실행 파일** 지원

## Project Structure

```
2025_iamsec/
├─ dtms/                     # 라이브러리 (핵심 로직)
│  ├─ init.py
│  ├─ converter.py           # 기존 dtms_to_csv.py 로직 이관 (함수화)
│  └─ cli.py                 # CLI 엔트리 포인트 (콘솔용)
├─ app/
│  └─ streamlit_app.py       # 웹 UI (Streamlit)
├─ scripts/
│  ├─ build_binary.sh        # PyInstaller 빌드 스크립트(맥/리눅스)
│  └─ run_app.sh             # Streamlit 실행 스크립트
├─ pyproject.toml            # 패키징/설치 (콘솔 명령 등록)
├─ requirements.txt
├─ .gitignore
└─ README.md
```


---
## ⚙️ Installation

```bash
# 1. 클론
git clone https://github.com/Donquixote95/2025_iamsec.git
cd 2025_iamsec

# 2. 가상환경 생성 & 라이브러리 설치
python -m venv .venv
source .venv/bin/activate   # (맥/리눅스)
# .venv\Scripts\activate    # (윈도우 PowerShell)

pip install -r requirements.txt
---

# 설치

pip install .

# 실행

dtms-to-csv --input input_sample.pdf --output output.csv --brigade "북극성여단"
