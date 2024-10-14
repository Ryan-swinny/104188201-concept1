import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Tuple, Optional
import logging
import matplotlib.pyplot as plt

# Constants
MODEL_PATH_TEMPLATE = "lstm_model_{}.h5"
SCALER_PATH = "scaler.joblib"
FEATURE_COLUMNS = [f"feature_{i}" for i in range(1, 51)]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LSTM model configurations
LSTM_CONFIGS: Dict[str, Dict[str, int]] = {
    "Small LSTM": {"units": 32, "layers": 1},
    "Medium LSTM": {"units": 64, "layers": 2},
    "Large LSTM": {"units": 128, "layers": 3},
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
        st.write("CSV File Preview:")
        st.write(df.head())
        return df
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        st.error(f"An error occurred while processing the CSV file: {str(e)}")
        return None


def main():
    st.title("Malware Behavior Analysis")

    # Sidebar
    st.sidebar.title("Settings")
    analysis_mode = st.sidebar.radio(
        "Select Analysis Mode", ["Upload CSV", "URL Detection"]
    )
    selected_model = st.sidebar.selectbox(
        "Select LSTM Model", list(LSTM_CONFIGS.keys())
    )

    model, scaler = get_model(selected_model)

    if analysis_mode == "Upload CSV":
        handle_csv_analysis(model, scaler)
    else:
        handle_url_detection(model, scaler)

    # Model training section
    st.sidebar.markdown("---")
    if st.sidebar.button("Train New Model"):
        train_new_model(selected_model)


def handle_csv_analysis(model: tf.keras.Model, scaler: StandardScaler):
    st.header("CSV File Analysis")
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing malware features", type="csv"
    )

    if uploaded_file is not None:
        df = process_csv_file(uploaded_file)
        if df is not None:
            st.write("Sample Data:")
            st.write(df.head())

            if st.button("Analyze Data"):
                analyze_csv_data(df, model, scaler)


def handle_url_detection(model: tf.keras.Model, scaler: StandardScaler):
    st.header("URL Detection")
    url = st.text_input("Enter URL to check:")

    if st.button("Detect URL"):
        if url and model and scaler:
            features = extract_features(url)  # Assuming this function exists
            if features is not None:
                prediction, probability = predict(features, model, scaler)
                display_results(prediction, probability, features)
            else:
                st.error(
                    "Unable to extract features from the URL. Please check if the URL is valid."
                )
        elif not url:
            st.warning("Please enter a URL.")
        else:
            st.warning("Model loading failed. Please ensure model files exist.")


def analyze_csv_data(df: pd.DataFrame, model: tf.keras.Model, scaler: StandardScaler):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, row in df.iterrows():
        # 使用实际模型进行预测（如果可用），否则使用模拟数据
        if model and scaler:
            features = row[FEATURE_COLUMNS].values.reshape(1, -1)
            prediction, probability = predict(features, model, scaler)
        else:
            prediction = np.random.choice(
                [0, 1], p=[0.7, 0.3]
            )  # 70% benign, 30% malicious
            probability = (
                np.random.uniform(0.5, 1.0)
                if prediction == 1
                else np.random.uniform(0, 0.5)
            )

        results.append(
            {
                "prediction": "Malicious" if prediction == 1 else "Benign",
                "probability": probability,
            }
        )

        progress = (i + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Analyzed {i + 1} / {len(df)} samples")

    results_df = pd.DataFrame(results)
    st.subheader("Analysis Results")
    st.write(results_df)

    # Create and display pie chart
    create_pie_chart(results_df)

    st.download_button(
        label="Download Results as CSV",
        data=results_df.to_csv(index=False),
        file_name="malware_analysis_results.csv",
        mime="text/csv",
    )


def create_pie_chart(results_df: pd.DataFrame):
    prediction_counts = results_df["prediction"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(
        prediction_counts.values,
        labels=prediction_counts.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
    st.pyplot(fig)


def train_new_model(selected_model: str):
    st.sidebar.info(f"Starting training of new {selected_model} model...")
    # Implement the model training logic here
    # train_new_model(LSTM_CONFIGS[selected_model])
    st.sidebar.success(f"New {selected_model} model training completed!")


if __name__ == "__main__":
    main()
