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

# **載入數據**（測試數據 & 訓練數據）
@st.cache_data
def load_data():
    # 替換為你的訓練數據和測試數據路徑
    #train_data = pd.read_csv("C:\\Users\\a0986\\Streamlit_Project\\Koching\\train_data.csv")
    #test_data = pd.read_csv("C:\\Users\\a0986\\Streamlit_Project\\Koching\\test_data.csv")
    x_train = pd.read_csv("x_train_subset[4].csv", index_col=0)
    y_train = pd.read_csv("y_train_subset[4].csv", index_col=0)
    x_test = pd.read_csv("x_test_subset[4].csv", index_col=0)
    y_test = pd.read_csv("y_test_subset[4].csv", index_col=0)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()  
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

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
    6: "6 Higher than master’s degree"
    }
    # **輸入特徵數據**
    st.sidebar.write("### Input Features")
        
    age = st.sidebar.number_input(label='Age', value=1, max_value=120, min_value=0, step=1)
    
    # 顯示選單（顯示文字）
    selected_education_label = st.sidebar.selectbox(
    "Education",
    options=list(education_options.values())
    )
    # 對應回數字（模型輸入用）
    education = [k for k, v in education_options.items() if v == selected_education_label][0]
    g_hei = st.sidebar.number_input(label='Height', value=180.0, step=0.1)
    g_wei = st.sidebar.number_input(label='Weight', value=40.0, min_value=40.0, step=0.1)
    lf_ldh = st.sidebar.number_input(label='LDH', value=0.0, step=0.1)
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
            st.write(f"Predicted Probability of Class 1 (有肺功能異常): {predicted_probability:.2f}")
            

            # **校準模型**
            calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            calibrated_model.fit(x_train, y_train)
            probs_calibrated = calibrated_model.predict_proba(x_test)[:, 1]
            
            # Lose 和 Win 分佈
            lose_likelihood_cal = probs_calibrated[y_test == 0]
            win_likelihood_cal = probs_calibrated[y_test == 1]

            # 計算 ROC 曲線和最佳閾值
            fpr, tpr, thresholds = roc_curve(y_test, probs_calibrated)
            j_scores = tpr - fpr
            optimal_idx = j_scores.argmax()
            optimal_threshold = thresholds[optimal_idx]
            if predicted_probability >= optimal_threshold:
                st.error("The model predicts: 有肺功能異常")
            else:
                st.success("The model predicts: 無肺功能異常")
            # **繪製機率分佈圖**
            sns.set(style="darkgrid")
            plt.figure(figsize=(12, 6))
            sns.kdeplot(lose_likelihood_cal, color="blue", shade=True, label="False Likelihood (Calibrated)")
            sns.kdeplot(win_likelihood_cal, color="red", shade=True, label="True Likelihood (Calibrated)")
            plt.axvline(optimal_threshold, color="black", linestyle="--", label=f"Threshold: {optimal_threshold:.2f}")
            plt.axvline(predicted_probability, color="red", linestyle="--", label=f"Predict probability: {predicted_probability:.2f}")
            #plt.axvline(0.5, color="black", linestyle="--", label="Threshold: 0.5")
            plt.title("Likelihood Distribution (0~1 Range)", fontsize=16)
            plt.xlim(0,1)
            plt.xlabel("Predicted Likelihood", fontsize=14)
            plt.ylabel("Density", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)

            # **顯示分佈圖**

            st.subheader("Likelihood Distribution with Threshold")
            st.pyplot(plt)
            st.subheader("Decision Tree Visualization")
            st.image("tree4_20250226.png")

        except ValueError as e:
            st.error(f"An error occurred: {e}")

# **顯示主頁面**
if __name__ == '__main__':
    show_predict_page()
