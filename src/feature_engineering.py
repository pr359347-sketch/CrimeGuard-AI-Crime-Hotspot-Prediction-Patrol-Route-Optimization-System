import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass

    def ingest_weather_data(self, df, weather_df):
        """
        Merges weather data with crime datasets.
        Crucially, ensure timestamps align without duplicating rows.
        """
        return df.merge(weather_df.groupby('timestamp').first(), on='timestamp', how='left')

    def ingest_event_data(self, df, event_df):
        """
        Adds binary feature for local events.
        """
        df = df.merge(event_df.groupby('timestamp').first(), on='timestamp', how='left')
        df['is_event'] = df['event_type'].notnull().astype(int) if 'event_type' in df.columns else 0
        return df

    def ingest_socioeconomic_data(self, df, socio_df):
        """
        Merges static socioeconomic features with crime datasets.
        Ensure one-to-one mapping per region_id.
        """
        return df.merge(socio_df.groupby('region_id').first(), on='region_id', how='left')

    def run_pipeline(self, crime_df, weather_df, event_df, socio_df=None):
        df = self.ingest_weather_data(crime_df, weather_df)
        df = self.ingest_event_data(df, event_df)
        if socio_df is not None:
            df = self.ingest_socioeconomic_data(df, socio_df)
        # Ensure numeric columns only before filling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df
