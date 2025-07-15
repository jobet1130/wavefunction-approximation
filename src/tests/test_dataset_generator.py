import os
import unittest

import pandas as pd

from src.dataprep.dataset_generator import generate_wavefunction_dataset


class TestWavefunctionDatasetGeneration(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = "tests/tmp_data"
        self.test_filename = "test_wavefunction_dataset.csv"
        self.csv_path = os.path.join(self.test_output_dir, self.test_filename)

        # Ensure clean test environment
        if not os.path.exists(self.test_output_dir):
            os.makedirs(self.test_output_dir)
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def test_generate_dataset(self):
        generate_wavefunction_dataset(
            potential="harmonic",
            num_samples=10,
            grid_points=64,
            noise=False,
            seed=0,
            output_dir=self.test_output_dir,
            filename=self.test_filename,
        )

        # Check if file was created
        self.assertTrue(os.path.exists(self.csv_path))

        # Load and check content
        df = pd.read_csv(self.csv_path)
        self.assertFalse(df.empty)
        self.assertIn("x", df.columns)
        self.assertIn("V(x)", df.columns)
        self.assertIn("psi(x)", df.columns)

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        if os.path.exists(self.test_output_dir):
            os.rmdir(self.test_output_dir)


if __name__ == "__main__":
    unittest.main()
