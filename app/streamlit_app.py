# app/streamlit_app.py
import streamlit as st
import pandas as pd
from dtms.converter import parse_pdf_to_df

st.set_page_config(page_title="IMA-SEC DTMS Parser", page_icon="📄", layout="centered")

st.title("📄 IMA-SEC DTMS Parser")
st.caption("PDF(주/세로 병합셀) → CSV 변환기 · IoU 기반 단일/공통 라벨 판정")

with st.container():
    col1, col2 = st.columns([2,1])
    with col1:
        uploaded = st.file_uploader("PDF 업로드", type=["pdf"])
    with col2:
        brigade = st.text_input("여단명 (선택)", value="북극성여단")

    run = st.button("변환 실행 🚀", use_container_width=True, type="primary")

if run:
    if not uploaded:
        st.warning("PDF를 업로드해주세요.")
    else:
        pdf_bytes = uploaded.read()
        with st.spinner("PDF 분석 중..."):
            df = parse_pdf_to_df(pdf_bytes, brigade=brigade.strip() or None)

        st.success(f"변환 완료! 총 {len(df)}행")
        if len(df) == 0:
            st.info("인식된 항목이 없습니다. PDF 텍스트 레이어/서식 또는 약어 사전을 확인하세요.")
        else:
            st.dataframe(df, use_container_width=True, height=420)
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "CSV 다운로드",
                data=csv,
                file_name="output.csv",
                mime="text/csv",
                use_container_width=True
            )

st.markdown("---")
with st.expander("⚙️ 도움말 / 주의사항"):
    st.markdown(
        """
        - 표의 '주' 헤더를 기준으로 각 주 열에 스냅합니다.  
        - 세로 병합된 공통 라벨(예: **혹한기훈련**)은 모든 유닛에,  
          단일 라벨(예: **WMD**)은 해당 유닛에만 배정됩니다.  
        - 라벨 표기가 다르면 `ACTIVITY_ALIASES`에 키를 추가하세요.
        """
    )