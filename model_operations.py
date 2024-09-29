import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
from LSTM_extraction import LSTMFeatureExtractor


@st.cache_resource
def get_model(model_type):
    try:
        if model_type == "LSTM":
            model = tf.keras.models.load_model(f"{model_type.lower()}_model.h5")
            lstm_extractor = joblib.load(f"{model_type.lower()}_extractor.joblib")
            return model, lstm_extractor
        else:
            model = joblib.load(f'{model_type.lower().replace(" ", "_")}_model.joblib')
            return model, None
    except Exception as e:
        st.error(f"Error loading {model_type} model: {str(e)}")
        return None, None


def predict_url(url, model, model_type, lstm_extractor=None):
    if model_type == "LSTM":
        lstm_features, additional_features = lstm_extractor.extract_combined_features(
            url
        )
        prediction = model.predict(
            [np.array([lstm_features]), np.array([additional_features])]
        )[0][0]
        return int(prediction > 0.5), prediction
    else:
        from feature_extraction import extract_features

        features = extract_features(url)
        if features is None:
            return None, None
        features_df = pd.DataFrame([features])
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0][1]
        return int(prediction), probability


def evaluate_model(model, X_test, y_test, model_type, lstm_extractor=None):
    if model_type == "LSTM":
        lstm_features = lstm_extractor.transform(X_test)
        additional_features = [
            lstm_extractor.extract_additional_features(url) for url in X_test
        ]
        return model.evaluate([lstm_features, np.array(additional_features)], y_test)[1]
    else:
        return model.score(X_test, y_test)
