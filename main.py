# main.py

import streamlit as st
import pandas as pd
import shap
import json
from catboost import CatBoostClassifier

from agent import build_agent_graph


# ----------------------
# ✅ 모델 & SHAP 로드
# ----------------------

model = CatBoostClassifier()
model.load_model("catboost_cnc_fault_model.cbm")

explainer = shap.TreeExplainer(model)

df = pd.read_csv('data/X_train_labeled.csv')
feature_names = df.columns.tolist()


agent_app = build_agent_graph()

# ----------------------
# ✅ Streamlit UI
# ----------------------

st.set_page_config(page_title="CNC AI Agent", layout="wide")
st.title("🧠 CNC 불량 원인 분석 AI Agent")

uploaded_file = st.file_uploader("📥 CNC 센서 데이터 업로드 (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/sample_cnc_input.csv")

st.dataframe(df.head())

sample_idx = st.number_input("분석할 샘플 인덱스", 0, len(df) - 1, 0)

if st.button("🚀 불량 분석 실행"):

    X = df.iloc[[sample_idx]]
    sensor_snapshot = df.iloc[sample_idx].to_dict()

    prob = model.predict_proba(X)[0, 1]

    if prob >= 0.6:
        st.subheader("⚠️ 불량입니다! 분석을 진행합니다.")
        with st.spinner("분석중..."):
            shap_vals = explainer.shap_values(X)[0]
            abs_shap = [abs(v) for v in shap_vals]
            top_idx = sorted(range(len(abs_shap)), key=lambda i: abs_shap[i], reverse=True)[:5]

            shap_top_features = [
                {"feature": feature_names[i], "value": float(shap_vals[i])} for i in top_idx
            ]

            init_state = {
                "fault_prob": float(prob),
                "shap_top_features": shap_top_features,
                "sensor_snapshot": sensor_snapshot,
            }

            final_state = agent_app.invoke(init_state)

            st.subheader("✅ 불량 예측 결과")
            st.metric("불량 확률", f"{prob:.2f}")

            st.subheader("📌 SHAP 주요 변수")
            st.json(shap_top_features)

            st.subheader("🧠 Cause Agent 결과")
            st.json(final_state["cause_result"])

            st.subheader("📄 RAG 조치 가이드")
            st.text_area("조치 가이드", final_state["rag_context"], height=300)

            st.subheader("📑 최종 자동 리포트")
            st.text_area("최종 리포트", final_state["final_answer"], height=350)
    else:
        st.subheader("🟢정상입니다.")
