# 🧠 Wavefunction Approximation using Machine Learning

This project explores how machine learning, particularly neural networks built with Keras, can approximate quantum mechanical wavefunctions by learning the solutions to the time-independent Schrödinger equation.

---

## 🧪 Problem Statement

In quantum mechanics, the time-independent Schrödinger equation describes the energy states of a particle in a potential field:

\[
\hat{H} \psi(x) = E \psi(x)
\]

where:

- \( \hat{H} = -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + V(x) \) is the Hamiltonian operator  
- \( \psi(x) \) is the wavefunction  
- \( V(x) \) is the potential energy  
- \( E \) is the energy eigenvalue  

The goal is to approximate \( \psi(x) \) using neural networks under various quantum potentials (e.g., harmonic oscillator, square well).

---

## 🎯 Objectives

- Approximate ground-state wavefunctions \( \psi(x) \) using supervised learning.
- Incorporate physical laws using **Physics-Informed Neural Networks (PINNs)**.
- Estimate the ground state energy \( E \) through **Variational Monte Carlo (VMC)**.

---

## 🧠 Techniques Used

### ✅ Supervised Learning
Train a Keras-based MLP to match analytical \( \psi(x) \) from known systems (e.g., harmonic oscillator).

### ✅ Physics-Informed Neural Networks (PINNs)
Define a custom loss function that penalizes deviations from the Schrödinger equation:

\[
\mathcal{L}_{\text{PINN}} = \left\| \hat{H} \psi(x) - E \psi(x) \right\|^2
\]

This forces the model to obey physical constraints during training.

### ✅ Variational Monte Carlo (VMC)
Use neural networks (e.g., dense or RBM) as variational wavefunctions, then optimize expected energy:

\[
E[\psi] = \frac{\langle \psi | \hat{H} | \psi \rangle}{\langle \psi | \psi \rangle}
\]

using stochastic sampling.

---

## ⚙️ Key Components

- Keras-based architectures for MLP, PINNs, and NQS
- Exact solution generators for training data
- Schrödinger operator applied using `tf.GradientTape`
- Custom loss functions for physics-informed optimization
- Metropolis sampling for energy-based training

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/jobet1130/wavefunction-approximation.git
   cd wavefunction-approximation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter:
   ```bash
   jupyter lab
   ```

4. Start with `01_data_generation.ipynb`, then move to `02_supervised_mlp_keras.ipynb` or `03_pinn_keras.ipynb`.

---

## 📦 Requirements

- Python 3.9+
- TensorFlow ≥ 2.13 (Keras API)
- NumPy, SciPy, Matplotlib
- JupyterLab

See `requirements.txt` for the full list.

---

## 📈 Outputs

- Approximated wavefunctions \( \psi(x) \)
- Estimated energy levels \( E \)
- Loss convergence plots
- Schrödinger residuals \( H\psi - E\psi \)

---

## 🧾 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by PINNs research and deep learning approaches to quantum mechanics.
- Uses TensorFlow Keras for model construction and training.
- Built with ❤️ for physics and machine learning.