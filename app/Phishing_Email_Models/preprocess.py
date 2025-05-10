import pandas as pd
import numpy as np

def preprocess_data(df):
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df
