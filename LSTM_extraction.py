import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from urllib.parse import urlparse


class LSTMFeatureExtractor:
    def __init__(self, max_url_length=100, vocab_size=10000):
        # 初始化特徵提取器
        self.max_url_length = max_url_length
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(num_words=vocab_size, char_level=True)

    def fit_transform(self, urls):
        # 訓練tokenizer並轉換URL為序列
        self.tokenizer.fit_on_texts(urls)
        sequences = self.tokenizer.texts_to_sequences(urls)
        return pad_sequences(sequences, maxlen=self.max_url_length)

    def transform(self, urls):
        # 將新的URL轉換為序列
        sequences = self.tokenizer.texts_to_sequences(urls)
        return pad_sequences(sequences, maxlen=self.max_url_length)

    def extract_additional_features(self, url):
        # 提取URL的額外特徵
        parsed = urlparse(url)
        return {
            "length": len(url),
            "domain_length": len(parsed.netloc),
            "path_length": len(parsed.path),
            "num_params": len(parsed.query.split("&")) if parsed.query else 0,
            "num_digits": sum(c.isdigit() for c in url),
            "num_special_chars": len([c for c in url if not c.isalnum()]),
            "is_https": 1 if parsed.scheme == "https" else 0,
            "num_subdomains": len(parsed.netloc.split(".")) - 1,
        }

    def extract_combined_features(self, url):
        # 結合LSTM特徵和額外特徵
        lstm_features = self.transform([url])[0]
        additional_features = self.extract_additional_features(url)
        return lstm_features, list(additional_features.values())


def create_lstm_model(vocab_size, max_url_length, num_additional_features):
    # 創建LSTM模型
    model = Sequential(
        [
            Embedding(vocab_size, 32, input_length=max_url_length),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    # 添加額外特徵輸入
    additional_input = tf.keras.layers.Input(shape=(num_additional_features,))
    concat = tf.keras.layers.Concatenate()([model.output, additional_input])
    output = tf.keras.layers.Dense(1, activation="sigmoid")(concat)

    # 組合LSTM和額外特徵
    combined_model = tf.keras.Model(
        inputs=[model.input, additional_input], outputs=output
    )
    combined_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    return combined_model
