import os

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from tqdm import tqdm


def infinite_square_well(x, L=1.0):
    return np.where((x >= 0) & (x <= L), 0, np.inf)


def harmonic_oscillator(x, m=1.0, omega=1.0):
    return 0.5 * m * omega**2 * x**2


def finite_square_well(x, V0=50.0, a=1.0):
    return np.where(np.abs(x) <= a, -V0, 0)


def ground_state_wavefunction(potential_type, x, **kwargs):
    if potential_type == "infinite_well":
        L = kwargs.get("L", 1.0)
        psi = np.where((x >= 0) & (x <= L), np.sqrt(2 / L) * np.sin(np.pi * x / L), 0)
    elif potential_type == "harmonic":
        m = kwargs.get("m", 1.0)
        omega = kwargs.get("omega", 1.0)
        hbar = kwargs.get("hbar", 1.0)
        alpha = np.sqrt(m * omega / hbar)
        psi = (alpha / np.pi**0.25) * np.exp(-0.5 * (alpha * x) ** 2)
    elif potential_type == "finite_well":
        psi = np.exp(-np.abs(x))
        psi /= np.sqrt(trapezoid(psi**2, x))
    else:
        raise ValueError(f"Unknown potential type: {potential_type}")
    return psi


def generate_wavefunction_dataset(
    potential="harmonic",
    num_samples=5000,
    grid_points=128,
    noise=False,
    seed=42,
    output_dir="../../data/raw",
    filename="wavefunction_dataset.csv",
    **kwargs,
):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if potential == "infinite_well":
        L = kwargs.get("L", 1.0)
        x = np.linspace(0, L, grid_points)
    else:
        x = np.linspace(-5, 5, grid_points)

    records = []

    for _ in tqdm(range(num_samples), desc=f"Generating {num_samples} samples"):
        if potential == "harmonic":
            V = harmonic_oscillator(x, **kwargs)
        elif potential == "infinite_well":
            V = infinite_square_well(x, **kwargs)
        elif potential == "finite_well":
            V = finite_square_well(x, **kwargs)
        else:
            raise ValueError(f"Unsupported potential type: {potential}")

        psi = ground_state_wavefunction(potential, x, **kwargs)

        if noise:
            psi += np.random.normal(scale=0.01, size=psi.shape)

        psi /= np.sqrt(trapezoid(psi**2, x))

        for xi, Vi, psii in zip(x, V, psi):
            records.append([xi, Vi, psii])

    df = pd.DataFrame(records, columns=["x", "V(x)", "psi(x)"])

    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)

    print(f"✅ Dataset saved: {csv_path}")


if __name__ == "__main__":
    generate_wavefunction_dataset(
        potential="harmonic",
        num_samples=5000,
        grid_points=128,
        noise=True,
        seed=42,
        output_dir="../../data/raw",
    )
