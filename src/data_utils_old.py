import pandas as pd
import numpy as np

from pathlib import Path
from typing import Tuple, Optional

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from configs.config import CRIME, WEATHER, TRAINING
from utils.logger import logger
from utils.paths import RAW_DATA, PROCESSED_DATA


class DataProcessor:

    def __init__(self):

        self.scaler = MinMaxScaler()

        self.raw_path = RAW_DATA

        self.processed_path = PROCESSED_DATA

        logger.info("Data Processor Initialized")


    ##############################################################
    # LOAD DATA
    ##############################################################

    def load_crime_data(self):

        file_path = self.raw_path / CRIME["file_name"]

        logger.info(f"Loading Crime Dataset : {file_path}")

        crime = pd.read_csv(
            file_path,
            low_memory=False
        )

        logger.info(f"Crime Shape : {crime.shape}")

        return crime


    def load_weather_data(self):

        file_path = self.raw_path / WEATHER["file_name"]

        if not file_path.exists():

            logger.warning("Weather Dataset Not Found")

            return None

        logger.info(f"Loading Weather Dataset : {file_path}")

        weather = pd.read_csv(file_path)

        logger.info(f"Weather Shape : {weather.shape}")

        return weather


    ##############################################################
    # CLEANING
    ##############################################################

    def standardize_columns(self, df):

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("/", "_")
        )

        return df


    def clean_crime_data(self, df):

        logger.info("Cleaning Crime Dataset...")

        df = self.standardize_columns(df)

        if "date" in df.columns:

            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

        df = df.drop_duplicates()

        df = df.dropna(
            subset=[
                "latitude",
                "longitude"
            ],
            how="any"
        )

        logger.info(f"Clean Shape : {df.shape}")

        return df
    
        ##############################################################
    # FEATURE ENGINEERING
    ##############################################################

    def prepare_features(self, df):

        logger.info("Preparing Features...")

        drop_columns = [
            "id",
            "case_number",
            "location",
            "block",
            "iucr",
            "description",
            "updated_on",
            "fbi_code"
        ]

        existing = [c for c in drop_columns if c in df.columns]

        df = df.drop(columns=existing)

        # Boolean → Integer
        if "arrest" in df.columns:
            df["arrest"] = df["arrest"].astype(int)

        if "domestic" in df.columns:
            df["domestic"] = df["domestic"].astype(int)

        # Encode categorical columns
        categorical = df.select_dtypes(include=["object"]).columns

        for col in categorical:

            if col == "date":
                continue

            df[col] = df[col].astype("category").cat.codes

        # Time features
        if "date" in df.columns:

            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day
            df["hour"] = df["date"].dt.hour
            df["dayofweek"] = df["date"].dt.dayofweek

            df = df.drop(columns=["date"])

        logger.info("Feature Engineering Completed")

        return df


    ##############################################################
    # SCALE FEATURES
    ##############################################################

    def scale_features(self, df):

        numeric_cols = df.select_dtypes(include=np.number).columns

        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        logger.info("Scaling Completed")

        return df


    ##############################################################
    # TRAIN / TEST SPLIT
    ##############################################################

    def split_dataset(self, df):

        X = df.iloc[:, :-1].values

        y = df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42,

            shuffle=False

        )

        logger.info(f"Train : {X_train.shape}")

        logger.info(f"Test  : {X_test.shape}")

        return X_train, X_test, y_train, y_test


    ##############################################################
    # DATALOADER
    ##############################################################

    def create_dataloader(

        self,

        X,

        y,

        batch_size=None,

        sequence_length=7

    ):

        if batch_size is None:
            batch_size = TRAINING["batch_size"]

        dataset = CrimeDataset(

            X,

            y,

            sequence_length

        )

        loader = DataLoader(

            dataset,

            batch_size=batch_size,

            shuffle=True,

            drop_last=False

        )

        logger.info("DataLoader Created")

        return loader
    

        ##############################################################
    # SAVE DATA
    ##############################################################

    def save_processed_data(
        self,
        df,
        filename="crime_processed.csv"
    ):

        save_path = self.processed_path / filename

        df.to_csv(save_path, index=False)

        logger.info(f"Processed dataset saved -> {save_path}")

        return save_path


    ##############################################################
    # COMPLETE PIPELINE
    ##############################################################

    def run_pipeline(self):

        logger.info("=" * 60)
        logger.info("Starting CrimeGuard Data Pipeline")
        logger.info("=" * 60)

        crime = self.load_crime_data()

        crime = self.clean_crime_data(crime)

        crime = self.prepare_features(crime)

        crime = self.scale_features(crime)

        self.save_processed_data(crime)

        X_train, X_test, y_train, y_test = self.split_dataset(crime)

        train_loader = self.create_dataloader(
            X_train,
            y_train
        )

        test_loader = self.create_dataloader(
            X_test,
            y_test
        )

        logger.info("=" * 60)
        logger.info("Pipeline Completed Successfully")
        logger.info("=" * 60)

        return {
            "processed_data": crime,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "train_shape": X_train.shape,
            "test_shape": X_test.shape
        }


##############################################################
# MAIN
##############################################################

if __name__ == "__main__":

    processor = DataProcessor()

    results = processor.run_pipeline()

    print("\n")
    print("=" * 60)
    print("CrimeGuard AI Data Pipeline")
    print("=" * 60)
    print(f"Processed Shape : {results['processed_data'].shape}")
    print(f"Train Shape     : {results['train_shape']}")
    print(f"Test Shape      : {results['test_shape']}")
    print("=" * 60)