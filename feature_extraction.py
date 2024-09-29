from urllib.parse import urlparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def extract_features(url):
    try:
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
    except Exception as e:
        print(f"Error extracting features from URL: {url}. Error: {str(e)}")
        return None


def load_and_prepare_data(data):
    print("Preparing data...")

    # Check if input is a file path or DataFrame
    if isinstance(data, str):
        print("Loading data from file...")
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        print("Using provided DataFrame...")
        df = data.copy()
    else:
        raise ValueError("Input must be a file path (string) or a pandas DataFrame")

    # Check if required columns exist
    required_columns = ["url", "type"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"DataFrame must contain the following columns: {required_columns}"
        )

    print("Extracting features...")
    X = df["url"].apply(extract_features).apply(pd.Series)
    y = df["type"]

    # Remove rows with None values (failed feature extraction)
    valid_rows = X.notna().all(axis=1)
    X = X[valid_rows]
    y = y[valid_rows]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
