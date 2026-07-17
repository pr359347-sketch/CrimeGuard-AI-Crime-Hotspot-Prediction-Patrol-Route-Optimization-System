import pandas as pd

def show_shape(df, name="Dataset"):
    print(f"{name} Shape : {df.shape}")

def show_columns(df):
    print(df.columns.tolist())

def missing_values(df):
    return df.isnull().sum()

def duplicate_rows(df):
    return df.duplicated().sum()