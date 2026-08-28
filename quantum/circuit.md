Qubit 0: ──[ H ]──[ RY(x₀) ]──╭●──────────╭X──[ RY(θ₀) ]── ... ──┤ ⟨Z₀⟩
Qubit 1: ──[ H ]──[ RY(x₁) ]──╰X──╭●──────│───[ RY(θ₁) ]── ... ──┤ ⟨Z₁⟩
Qubit 2: ──[ H ]──[ RY(x₂) ]──────╰X──╭●──│───[ RY(θ₂) ]── ... ──┤ ⟨Z₂⟩
Qubit 3: ──[ H ]──[ RY(x₃) ]──────────╰X──╰●──[ RY(θ₃) ]── ... ──┤ ⟨Z₃⟩
          [ State Embedding ] [ 2-Qubit Entangler ] [ Trainable ]   [ Measurement ]

#### A. Embedding Layer Gates (State Encoding)
1. **Hadamard Gate ($H$)**:
   * Placed on all 4 qubits to create an equal superposition state $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$.
   * Matrix:
     $$H = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1 \\ 1 & -1\end{bmatrix}$$
2. **Feature Rotation Gate ($R_y(x_i)$)**:
   * Rotates each qubit along the Y-axis by the classical feature angle $x_i \in [-\pi, \pi]$ from the classical backbone.
   * Matrix:
     $$R_y(x_i) = \begin{bmatrix}\cos\frac{x_i}{2} & -\sin\frac{x_i}{2} \\ \sin\frac{x_i}{2} & \cos\frac{x_i}{2}\end{bmatrix}$$

---

#### B. Variational Layer Gates (Trainable Processing, repeated 4 times)
3. **Controlled-NOT Entangling Gate ($\text{CNOT}$)**:
   * 2-qubit entangling gates applied in a circular ring topology:
     $$\text{CNOT}(0 \rightarrow 1), \quad \text{CNOT}(1 \rightarrow 2), \quad \text{CNOT}(2 \rightarrow 3), \quad \text{CNOT}(3 \rightarrow 0)$$
   * Generates quantum entanglement across fundus feature dimensions:
     $$\text{CNOT} = \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0\end{bmatrix}$$
4. **Trainable Parameterized Rotation Gates ($R_y(\theta_{l, i})$)**:
   * Single-qubit rotations with 16 trainable variational angles $\theta_{l, i}$ (4 layers $\times$ 4 qubits) optimized via backpropagation and Adam optimizer.

---

#### C. Measurement Layer Observables
5. **Pauli-$Z$ Expectation ($\langle \sigma_z \rangle$)**:
   * Measures the projection of each qubit along the Z-axis:
     $$\sigma_z = \begin{bmatrix}1 & 0 \\ 0 & -1\end{bmatrix}, \quad \langle Z_i \rangle = \langle \psi | \sigma_z^{(i)} | \psi \rangle \in [-1, 1]$$
   * The resulting 4-dimensional continuous vector $\left[\langle Z_0 \rangle, \langle Z_1 \rangle, \langle Z_2 \rangle, \langle Z_3 \rangle\right]$ is fed into the linear classification head for the 5 Diabetic Retinopathy stages.

---

### 🎛️ 2. Other Supported Gate Combinations (Table 5 Presets)

You can switch to any of the other gate variations explored in the paper by passing `--embedding` and `--entangling` flags:

| Configuration | Embedding Gates | Entangling Gates | Command Argument |
|---|---|---|---|
| **Default (Paper baseline)** | $H + R_y(x)$ | $\text{CNOT}$ | `--embedding hadamard --entangling cnot` |
| **Phase Variation** | $S + H + R_y(x)$ | $\text{CNOT}$ | `--embedding s_phase --entangling cnot` |
| **Adjoint Phase** | $S^\dagger + H + R_y(x)$ | $\text{CNOT}$ | `--embedding s_dagger --entangling cnot` |
| **X-Rotation Embedding** | $R_x(x)$ | $\text{CNOT}$ | `--embedding rx --entangling cnot` |
| **Controlled-Z** | $H + R_y(x)$ | $\text{CZ}$ | `--embedding hadamard --entangling cz` |
| **SWAP Entanglement** | $H + R_y(x)$ | $\text{SWAP}$ | `--embedding hadamard --entangling swap` |
| **Controlled-RX** | $H + R_y(x)$ | $\text{CR}_x(\theta)$ | `--embedding hadamard --entangling crx` |
| **Dual Rotation** | $R_x(x)$ | $\text{CR}_x(\theta)$ | `--embedding rx --entangling crx` |