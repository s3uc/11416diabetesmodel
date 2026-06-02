import streamlit as st
import pandas as pd
import joblib

# 모델과 스케일러 불러오기
knn_model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 당뇨병 예측 웹앱")

st.write("건강 정보를 입력하면 당뇨병 여부를 예측합니다.")

# 입력 받기
preg = st.number_input("임신 횟수", min_value=0, value=1)
glucose = st.number_input("혈당", min_value=0.0, value=120.0)
bp = st.number_input("혈압", min_value=0.0, value=70.0)
skin = st.number_input("피부 두께", min_value=0.0, value=20.0)
insulin = st.number_input("인슐린 수치", min_value=0.0, value=80.0)
bmi = st.number_input("체질량지수 (BMI)", min_value=0.0, value=25.0)
gene = st.number_input("당뇨 유전 지수", min_value=0.0, value=0.5)
age = st.number_input("나이", min_value=0, value=30)

# 예측 버튼
if st.button("당뇨 예측하기"):

    # DataFrame 생성
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, gene, age]],
        columns=[
            '임신 횟수', '혈당', '혈압', '피부 두께',
            '인슐린 수치', '체질량지수',
            '당뇨 유전 지수', '나이'
        ]
    )

    # 스케일링
    input_scaled = scaler.transform(input_data)

    # 예측
    predicted = knn_model.predict(input_scaled)
    prob = knn_model.predict_proba(input_scaled)

    # 결과 출력
    if predicted[0] == 1:
        st.error("⚠️ 예측 결과: 당뇨")
    else:
        st.success("✅ 예측 결과: 정상")

    st.write(f"당뇨 확률: **{prob[0][1] * 100:.1f}%**")