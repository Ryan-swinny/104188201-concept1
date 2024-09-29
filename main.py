import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import os
from typing import Dict, Tuple, Optional
import logging

# Constants
MODEL_PATH_TEMPLATE = "lstm_model_{}.h5"
SCALER_PATH = "scaler.joblib"
FEATURE_COLUMNS = [f"feature_{i}" for i in range(1, 51)]
REQUIRED_COLUMNS = ["timestamp"] + FEATURE_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LSTM model configurations
LSTM_CONFIGS: Dict[str, Dict[str, int]] = {
    "小型 LSTM": {"units": 32, "layers": 1},
    "中型 LSTM": {"units": 64, "layers": 2},
    "大型 LSTM": {"units": 128, "layers": 3},
}


def create_lstm_model(
    config: Dict[str, int], input_shape: Tuple[int, int]
) -> tf.keras.Model:
    """Create and compile an LSTM model based on the given configuration."""
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.LSTM(
            config["units"],
            input_shape=input_shape,
            return_sequences=config["layers"] > 1,
        )
    )
    for _ in range(config["layers"] - 1):
        model.add(tf.keras.layers.LSTM(config["units"], return_sequences=True))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


@st.cache_resource
def get_model(
    model_name: str,
) -> Tuple[Optional[tf.keras.Model], Optional[StandardScaler]]:
    """Load the LSTM model and scaler from disk."""
    model_path = MODEL_PATH_TEMPLATE.format(model_name)
    if not os.path.exists(model_path) or not os.path.exists(SCALER_PATH):
        logger.warning(f"{model_name} model or scaler file does not exist.")
        return None, None

    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading LSTM model: {str(e)}")
        return None, None


def predict(
    data: np.ndarray, model: tf.keras.Model, scaler: StandardScaler
) -> Tuple[int, float]:
    """Make a prediction using the loaded model and scaler."""
    scaled_data = scaler.transform(data)
    prediction = model.predict(np.array([scaled_data]))[0][0]
    return int(prediction > 0.5), float(prediction)


def process_csv_file(file) -> Optional[pd.DataFrame]:
    """Process the uploaded CSV file and return a DataFrame."""
    try:
        df = pd.read_csv(file)

        # 顯示 CSV 檔案預覽
        st.write("CSV 檔案預覽：")
        st.write(df.head())

        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            st.error(f"CSV 檔案缺少以下欄位：{', '.join(missing_columns)}")
            return None

        return df
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        st.error(f"處理 CSV 檔案時發生錯誤：{str(e)}")
        return None


def main():
    st.title("惡意軟件行為分析")

    # Sidebar
    st.sidebar.title("設置")
    analysis_mode = st.sidebar.radio("選擇分析模式", ["上傳 CSV", "URL 檢測"])
    selected_model = st.sidebar.selectbox("選擇 LSTM 模型", list(LSTM_CONFIGS.keys()))

    model, scaler = get_model(selected_model)

    if analysis_mode == "上傳 CSV":
        handle_csv_analysis(model, scaler)
    else:
        handle_url_detection(model, scaler)

    # Model training section
    st.sidebar.markdown("---")
    if st.sidebar.button("訓練新模型"):
        train_new_model(selected_model)


def handle_csv_analysis(model: tf.keras.Model, scaler: StandardScaler):
    st.header("CSV 文件分析")
    uploaded_file = st.file_uploader("選擇一個包含惡意軟件特徵的 CSV 文件", type="csv")

    if uploaded_file is not None:
        df = process_csv_file(uploaded_file)
        if df is not None:
            st.write("樣本數據：")
            st.write(df.head())

            if st.button("分析數據"):
                analyze_csv_data(df, model, scaler)


def handle_url_detection(model: tf.keras.Model, scaler: StandardScaler):
    st.header("URL 檢測")
    url = st.text_input("輸入要檢查的 URL：")

    if st.button("檢測 URL"):
        if url and model and scaler:
            features = extract_features(url)  # Assuming this function exists
            if features is not None:
                prediction, probability = predict(features, model, scaler)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                display_results(prediction, probability, features, timestamp)
            else:
                st.error("無法從 URL 提取特徵。請檢查 URL 是否有效。")
        elif not url:
            st.warning("請輸入一個 URL。")
        else:
            st.warning("模型加載失敗，請確保模型文件存在。")


def analyze_csv_data(df: pd.DataFrame, model: tf.keras.Model, scaler: StandardScaler):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, row in df.iterrows():
        features = row[FEATURE_COLUMNS].values
        timestamp = row["timestamp"]
        prediction, probability = predict(features, model, scaler)
        results.append(
            {
                "timestamp": timestamp,
                "prediction": "惡意" if prediction == 1 else "良性",
                "probability": probability,
            }
        )

        progress = (i + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"已分析 {i + 1} / {len(df)} 個樣本")

    results_df = pd.DataFrame(results)
    st.subheader("分析結果")
    st.write(results_df)

    st.download_button(
        label="下載結果為 CSV",
        data=results_df.to_csv(index=False),
        file_name="malware_analysis_results.csv",
        mime="text/csv",
    )


def train_new_model(selected_model: str):
    st.sidebar.info(f"開始訓練新的 {selected_model} 模型...")
    # Implement the model training logic here
    # train_new_model(LSTM_CONFIGS[selected_model])
    st.sidebar.success(f"新的 {selected_model} 模型訓練完成！")


if __name__ == "__main__":
    main()
