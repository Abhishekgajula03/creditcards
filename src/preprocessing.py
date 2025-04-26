import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path):
    return pd.read_csv(path)


def preprocess(df):
    df = df.copy()
    df.fillna('Unknown', inplace=True)

    cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    num_cols = ['Age', 'Job', 'Credit amount', 'Duration']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    scaler = StandardScaler()

    X_cat = encoder.fit_transform(df[cat_cols])
    X_num = scaler.fit_transform(df[num_cols])

    import numpy as np
    X = np.hstack((X_num, X_cat))

    # For demonstration, generate binary target
    y = (df['Credit amount'] > df['Credit amount'].median()).astype(int)

    return train_test_split(X, y, test_size=0.2, random_state=42), encoder, scaler