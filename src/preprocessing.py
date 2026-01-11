import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df, target_col="Job_Suitable"):
    df = df.copy()

    encoders = {}
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, encoders
