import os
import unittest
import numpy as np
import pandas as pd

from src.dataprep.processor import WavefunctionProcessor


class TestWavefunctionProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_data_path = "data/raw/wavefunction_dataset.csv"
        if not os.path.exists(cls.sample_data_path):
            raise FileNotFoundError(f"Test dataset not found at {cls.sample_data_path}")
        cls.processor = WavefunctionProcessor(grid_points=128, scaling="minmax")

    def test_load_dataset(self):
        df = self.processor.load_dataset(self.sample_data_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn("x", df.columns)
        self.assertIn("V(x)", df.columns)
        self.assertIn("psi(x)", df.columns)

    def test_normalize(self):
        df = self.processor.load_dataset(self.sample_data_path)
        normalized = self.processor.normalize(df, ("x", "V(x)", "psi(x)"))
        self.assertIsInstance(normalized, pd.DataFrame)
        self.assertTrue((normalized.max() <= 1.0).all())
        self.assertTrue((normalized.min() >= 0.0).all())

    def test_reshape(self):
        df = self.processor.load_dataset(self.sample_data_path)
        normalized = self.processor.normalize(df, ("x", "V(x)", "psi(x)"))
        data = self.processor.reshape(normalized, ("x", "V(x)", "psi(x)"))
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape[1:], (128, 3))  # grid_points x 3 columns

    def test_preprocess(self):
        data, scaler = self.processor.preprocess(self.sample_data_path)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape[1:], (128, 3))
        self.assertIsNotNone(scaler)


if __name__ == "__main__":
    unittest.main()
