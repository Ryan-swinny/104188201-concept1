import numpy as np
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import streamlit as st

# 假設這個模塊包含了LSTM相關的特徵提取函數
from LSTM_extraction import LSTMFeatureExtractor, create_lstm_model

MODEL_CONFIGS = {
    "LSTM": {
        "小型": {"units": 32, "layers": 1},
        "中型": {"units": 64, "layers": 2},
        "大型": {"units": 128, "layers": 3},
    },
    "GRU": {
        "小型": {"units": 32, "layers": 1},
        "中型": {"units": 64, "layers": 2},
        "大型": {"units": 128, "layers": 3},
    },
}


def create_model(model_type, config, input_shape):
    model = Sequential()

    if model_type == "LSTM":
        model.add(
            LSTM(
                config["units"],
                input_shape=input_shape,
                return_sequences=config["layers"] > 1,
            )
        )
        for _ in range(config["layers"] - 1):
            model.add(LSTM(config["units"], return_sequences=True))
    elif model_type == "GRU":
        model.add(
            GRU(
                config["units"],
                input_shape=input_shape,
                return_sequences=config["layers"] > 1,
            )
        )
        for _ in range(config["layers"] - 1):
            model.add(GRU(config["units"], return_sequences=True))

    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(model_type, X_train, y_train, config=None):
    if model_type in ["LSTM", "GRU"]:
        return train_rnn_model(model_type, config, X_train, y_train)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def train_rnn_model(model_type, config, X_train, y_train):
    if model_type == "LSTM":
        lstm_extractor = LSTMFeatureExtractor()
        lstm_features = lstm_extractor.fit_transform(X_train)
        additional_features = [
            lstm_extractor.extract_additional_features(url) for url in X_train
        ]

        model = create_lstm_model(
            lstm_extractor.vocab_size,
            lstm_extractor.max_url_length,
            len(additional_features[0]),
        )
        history = model.fit(
            [lstm_features, np.array(additional_features)],
            y_train,
            epochs=config.get("epochs", 10),
            batch_size=config.get("batch_size", 32),
            validation_split=0.2,
        )
        return model, lstm_extractor, history
    else:  # GRU
        model = create_model(model_type, config, X_train.shape[1:])
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=config.get("epochs", 10),
            batch_size=config.get("batch_size", 32),
        )
        return model, None, history


# Streamlit 應用程序部分
def main():
    st.title("機器學習模型訓練系統")

    model_type = st.sidebar.selectbox(
        "選擇模型類型", ["Decision Tree", "Random Forest", "LSTM", "GRU"]
    )

    if model_type in ["LSTM", "GRU"]:
        model_size = st.sidebar.selectbox(
            "選擇模型大小", list(MODEL_CONFIGS[model_type].keys())
        )
        selected_config = MODEL_CONFIGS[model_type][model_size]
    else:
        selected_config = None

    # 這裡假設你有數據加載的功能
    X_train, y_train = load_data()

    if st.button("訓練模型"):
        with st.spinner("正在訓練模型..."):
            if model_type in ["LSTM", "GRU"]:
                model, extractor, history = train_model(
                    model_type, X_train, y_train, selected_config
                )
                st.success("模型訓練完成！")
                # 這裡可以添加顯示訓練歷史的代碼
            else:
                model = train_model(model_type, X_train, y_train)
                st.success("模型訓練完成！")

        # 這裡可以添加模型評估和結果展示的代碼


if __name__ == "__main__":
    main()
