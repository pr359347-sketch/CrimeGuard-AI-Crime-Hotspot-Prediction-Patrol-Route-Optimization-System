import pandas as pd
import numpy as np
import torch

from pathlib import Path

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from configs.config import CRIME, WEATHER, TRAINING
from utils.logger import logger
from utils.paths import RAW_DATA, PROCESSED_DATA


class CrimeDataset(Dataset):
    """
    Dataset class for CrimeGuard-AI
    """

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


class DataProcessor:

    def __init__(self):

        self.scaler = MinMaxScaler()

        self.encoder = LabelEncoder()

        self.raw_dir = RAW_DATA

        self.processed_dir = PROCESSED_DATA

        logger.info("DataProcessor Initialized")

    #######################################################
    # LOAD DATA
    #######################################################

    def load_crime_data(self):

        path = self.raw_dir / CRIME["file_name"]

        logger.info(f"Loading Crime Dataset : {path}")

        crime = pd.read_csv(
            path,
            low_memory=False
        )

        logger.info(f"Crime Shape : {crime.shape}")

        return crime

    def load_weather_data(self):

        path = self.raw_dir / WEATHER["file_name"]

        if not path.exists():

            logger.warning("Weather dataset not found.")

            return None

        weather = pd.read_csv(path)

        logger.info(f"Weather Shape : {weather.shape}")

        return weather

    #######################################################
    # CLEANING
    #######################################################

    def clean_crime_data(self, df):

        logger.info("Cleaning Crime Dataset...")

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("/", "_")
        )

        if "date" in df.columns:

            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

        df = df.drop_duplicates()

        if "latitude" in df.columns and "longitude" in df.columns:

            df = df.dropna(
                subset=[
                    "latitude",
                    "longitude"
                ]
            )

        logger.info(f"Shape After Cleaning : {df.shape}")

        return df
    
        #######################################################
    # FEATURE ENGINEERING
    #######################################################

    def prepare_features(self, df):

        logger.info("Preparing Features...")

        # Encode target column
        if "primary_type" in df.columns:
            df["crime_label"] = self.encoder.fit_transform(df["primary_type"])
            df = df.drop(columns=["primary_type"])

        # Boolean → Integer
        for col in ["arrest", "domestic"]:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Date Features
        if "date" in df.columns:
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day
            df["hour"] = df["date"].dt.hour
            df["dayofweek"] = df["date"].dt.dayofweek
            df = df.drop(columns=["date"])

        # Drop columns not useful for training
        drop_cols = [
            "id",
            "case_number",
            "block",
            "iucr",
            "description",
            "location_description",
            "location",
            "updated_on"
        ]

        existing = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing)

        # Encode remaining categorical columns
        object_cols = df.select_dtypes(include=["object", "string"]).columns

        for col in object_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        logger.info("Feature Engineering Completed")

        return df


    #######################################################
    # SCALING
    #######################################################

    def scale_data(self, df):

        logger.info("Scaling Features...")

        target = df["crime_label"]

        features = df.drop(columns=["crime_label"])

        features = self.scaler.fit_transform(features)

        features = pd.DataFrame(
            features,
            columns=df.drop(columns=["crime_label"]).columns
        )

        features["crime_label"] = target.values

        return features


    #######################################################
    # TRAIN TEST SPLIT
    #######################################################

    def split_data(self, df):

        X = df.drop(columns=["crime_label"]).values

        y = df["crime_label"].values

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )


    #######################################################
    # DATALOADER
    #######################################################

    def create_dataloader(self, X, y):

        dataset = CrimeDataset(X, y)

        loader = DataLoader(
            dataset,
            batch_size=TRAINING["batch_size"],
            shuffle=True
        )

        return loader
    
        #######################################################
    # SAVE DATA
    #######################################################

    def save_processed_data(self, df):

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        output_file = self.processed_dir / "crime_processed.csv"

        df.to_csv(output_file, index=False)

        logger.info(f"Processed data saved to {output_file}")

        return output_file


    #######################################################
    # COMPLETE PIPELINE
    #######################################################

    def run_pipeline(self):

        logger.info("========== Starting Data Pipeline ==========")

        crime_df = self.load_crime_data()

        weather_df = self.load_weather_data()

        # Weather integration (optional)
        if weather_df is not None:
            logger.info("Weather dataset loaded successfully.")
            # Future merge logic can be added here

        crime_df = self.clean_crime_data(crime_df)

        crime_df = self.prepare_features(crime_df)

        crime_df = self.scale_data(crime_df)

        self.save_processed_data(crime_df)

        X_train, X_test, y_train, y_test = self.split_data(crime_df)

        train_loader = self.create_dataloader(X_train, y_train)

        test_loader = self.create_dataloader(X_test, y_test)

        logger.info("========== Data Pipeline Completed ==========")

        return (
            train_loader,
            test_loader,
            X_train.shape,
            X_test.shape
        )


#######################################################
# MAIN
#######################################################

if __name__ == "__main__":

    processor = DataProcessor()

    train_loader, test_loader, train_shape, test_shape = processor.run_pipeline()

    print("\n========================================")
    print("Data Pipeline Completed Successfully")
    print("========================================")
    print(f"Train Shape : {train_shape}")
    print(f"Test Shape  : {test_shape}")
    print(f"Train Batches : {len(train_loader)}")
    print(f"Test Batches  : {len(test_loader)}")