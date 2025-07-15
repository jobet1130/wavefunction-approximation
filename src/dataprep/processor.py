import os
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class WavefunctionProcessor:
    def __init__(
        self,
        grid_points: int = 128,
        scaling: Literal["minmax", "standard", None] = "minmax"
    ):
        self.grid_points = grid_points
        self.scaling = scaling
        self.scaler = None

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Dataset not found at: {file_path}")
        return pd.read_csv(file_path)

    def normalize(
        self,
        df: pd.DataFrame,
        columns: Tuple[str, str, str]
    ) -> pd.DataFrame:
        if self.scaling is None:
            return df
        scaler_cls = MinMaxScaler if self.scaling == "minmax" else StandardScaler
        self.scaler = scaler_cls()
        scaled = self.scaler.fit_transform(
            df[list(columns)]
        )
        return pd.DataFrame(scaled, columns=columns)

    def reshape(
        self,
        df: pd.DataFrame,
        columns: Tuple[str, str, str]
    ) -> np.ndarray:
        records_per_sample = self.grid_points
        grouped = df.groupby(df.index // records_per_sample)
        samples = [
            group[list(columns)].to_numpy()
            for _, group in grouped
        ]
        return np.stack(samples)

    def preprocess(
        self,
        file_path: str,
        columns: Tuple[str, str, str] = (
            "x", "V(x)", "psi(x)"
        )
    ) -> Tuple[np.ndarray, Union[MinMaxScaler, StandardScaler, None]]:
        df = self.load_dataset(file_path)
        df = self.normalize(df, columns)
        data = self.reshape(df, columns)
        return data, self.scaler


if __name__ == "__main__":
    processor = WavefunctionProcessor(
        grid_points=128,
        scaling="minmax"
    )

    data_path = "../../data/raw/wavefunction_dataset.csv"
    try:
        data, scaler = processor.preprocess(data_path)
        print("✅ Preprocessing complete.")
        print("📐 Data shape:", data.shape)
        if scaler:
            print(f"📊 Used scaler: {scaler.__class__.__name__}")
    except FileNotFoundError as e:
        print(e)
