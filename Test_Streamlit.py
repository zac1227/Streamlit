import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.tree import plot_tree
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# **全局加載模型**（只執行一次）
@st.cache_resource  # 使用 Streamlit 的快取功能
def load_model():
    with open("decision_tree_model4.pkl", 'rb') as file:
        model = pickle.load(file)
    return model
    

# 加載模型
model = load_model()



# **主頁面函數**
def show_predict_page():
    st.title("ML Model Prediction")
    # 教育程度對應表
    education_options = {
    1: "1 Illiterate",
    2: "2 Primary school",
    3: "3 Junior high school",
    4: "4 Senior high school",
    5: "5 College & University",
    6: "6 Higher than master's degree"
    }
    # **輸入特徵數據**
    st.sidebar.write("### Input Features")
        
    age = st.sidebar.number_input(label='Age (18~90)', value=18, max_value=90, min_value=18, step=1)
    
    # 顯示選單（顯示文字）
    selected_education_label = st.sidebar.selectbox(
    "Education (Education attainment) (1: Illiterate, "
    "2: Primary school, "
    "3: Junior high school, "
    "4: Senior high school, "
    "5: College & University, "
    "6: Higher than master's degree)",
    options=list(education_options.values())
    )
    # 對應回數字（模型輸入用）
    education = [k for k, v in education_options.items() if v == selected_education_label][0]
    g_hei = st.sidebar.number_input(label='Height (cm)', value=180.0, step=0.1)
    g_wei = st.sidebar.number_input(label='Weight (kg)', value=40.0, min_value=40.0, step=0.1)
    lf_ldh = st.sidebar.number_input(label='LDH (Lactate dehydrogenase) (IU/L)', value=0.0, step=0.1)
    #cbc_leu = st.sidebar.number_input(label='cbc_leu', value=0.0,  step=0.1)
    #systolic = st.sidebar.number_input(label='systolic', value=0.0,  step=0.1)
    #lf_tb = st.sidebar.number_input(label='lf_tb', value=0.0,  step=0.1)
    
    cbc_leu=0
    systolic=0
    lf_tb=0
    #cbc_leu,systolic,lf_tb,
    # 收集用戶輸入數據
    input_data = [
    age,education,g_hei,lf_ldh,cbc_leu,systolic,lf_tb,g_wei
   
    ]

   

    X_input = np.array(input_data, dtype=float).reshape(1, -1)

    # **點擊 Predict 按鈕執行預測**
    if st.sidebar.button("Predict"):
        try:
            # 模型預測
            predicted_probability = model.predict_proba(X_input)[:, 1][0]
            predicted_class = model.predict(X_input)[0]

            # **顯示預測結果**
            st.subheader("Prediction Result")
            #st.write(f"Predicted Probability of Class 1 (有肺功能異常): {predicted_probability:.2f}")
            

            

            

            if predicted_class == 1:
                st.error("The model predicts: Positive")
            else:
                st.success("The model predicts: Negative")
           

            st.subheader("Decision Tree Visualization")
            st.image("tree4_20250420.png")

        except ValueError as e:
            st.error(f"An error occurred: {e}")

# **顯示主頁面**
if __name__ == '__main__':
    show_predict_page()
