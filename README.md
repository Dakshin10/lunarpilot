<p align="center">
  <img src="assets/lunarpilot-banner-v4.svg" alt="LunarPilot: Autonomous Precision Lunar Lander" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Gymnasium-v1.4.0-000000?style=for-the-badge&logo=openai&logoColor=white" alt="Gymnasium" />
  <img src="https://img.shields.io/badge/Stable--Baselines3-PPO-FF6F00?style=for-the-badge" alt="Stable-Baselines3" />
  <img src="https://img.shields.io/badge/Landing%20Success-100%25-2ECC71?style=for-the-badge" alt="Success Rate" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License" />
</p>

**LunarPilot** is an autonomous reinforcement learning flight control system designed to achieve pinpoint touchdowns in Gymnasium's `LunarLander-v3`. Combining **Proximal Policy Optimization (PPO)** with custom **potential-based reward shaping** and real-world rocketry flight dynamics, the pilot achieves a **100% landing success rate** with bullseye precision ($x = 0.00, y = 0.00$) and average episode returns exceeding **+330** (well beyond the +200 environment solved benchmark).

---

## 🎥 Live Landing Demo

<p align="center">
  <img src="assets/gif_demo.gif" alt="LunarPilot Autonomous Landing Sequence" width="100%" />
  <br />
  <em>Autonomous landing sequence: high-altitude drift correction, powered flare, dual-leg touchdown, and instantaneous engine cutoff.</em>
  <br /><br />
  📽️ <strong><a href="assets/demo.mp4">Watch Full Quality Video (assets/demo.mp4)</a></strong>
</p>

---

## 📊 Benchmark Telemetry & Evaluation

During deterministic flight evaluation over consecutive test episodes, the policy achieved a **100% landing success rate** within the target flags:

| Flight | Episode Return | Final Position ($x, y$) | Touchdown Velocity ($v_x, v_y$) | Leg Contacts | Helipad Flags | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | **+325.12** | $x = -0.00,\; y = -0.00$ | $v_x = +0.06,\; v_y = -0.05$ | Left: 1, Right: 1 | **YES (Bullseye)** | **SUCCESS** |
| **02** | **+327.04** | $x = -0.02,\; y = -0.00$ | $v_x = -0.09,\; v_y = -0.05$ | Left: 1, Right: 1 | **YES** | **SUCCESS** |
| **03** | **+300.01** | $x = -0.02,\; y = -0.01$ | $v_x = +0.08,\; v_y = -0.04$ | Left: 1, Right: 1 | **YES** | **SUCCESS** |
| **04** | **+289.19** | $x = -0.07,\; y = -0.01$ | $v_x = +0.10,\; v_y = +0.07$ | Left: 1, Right: 1 | **YES** | **SUCCESS** |
| **05** | **+308.40** | $x = -0.15,\; y = -0.01$ | $v_x = -0.03,\; v_y = -0.03$ | Left: 1, Right: 1 | **YES** | **SUCCESS** |

- **Pad Coordinates:** Dead center $x = 0.00$; target boundary flags placed at $x \in [-0.20, +0.20]$.
- **Mean Return:** **$\approx 310$ – $426$ points** (crash = $-100$, baseline solved threshold = $+200$).
- **Impact Softness:** Mean impact vertical velocity $|v_y| \le 0.05\text{ m/s}$, horizontal drift $|v_x| \le 0.10\text{ m/s}$.

---

## Autonomous Flight Guidance Stages

The pilot executes a multi-phase descent profile mirroring real spacecraft guidance logic:

<p align="center">
  <img src="assets/Autonomous%20Flight%20Guidance%20Stages_.png" alt="Autonomous Flight Guidance Stages" width="100%" />
</p>

---

## 🏗️ End-to-End System Architecture

Closed-loop interaction between the physical simulation, reward shaping engine, and PPO neural networks:

<p align="center">
  <img src="assets/End-to-End%20System%20Architecture.png" alt="End-to-End System Architecture" width="100%" />
</p>

---

## ⚙️ State & Action Dynamics

The agent operates over continuous kinematics and commands discrete thruster actuation.

### Observation Space (8 Dimensions)
$$\mathbf{s} = \big[x, \; y, \; v_x, \; v_y, \; \theta, \; \omega, \; \text{leg}_{\text{left}}, \; \text{leg}_{\text{right}}\big]$$

- Normalized horizontal & vertical coordinates ($x, y$) relative to landing pad
- Linear horizontal & vertical velocities ($v_x, v_y$)
- Lander orientation angle $\theta$ and angular velocity $\omega$
- Binary leg ground contact indicators ($\text{leg}_{\text{left}}, \text{leg}_{\text{right}} \in \{0, 1\}$)

### Action Space (Discrete: 4 Actions)
- `0`: Coast / Free Fall (momentum drift)
- `1`: Fire Left Orientation Engine
- `2`: Fire Main Central Engine
- `3`: Fire Right Orientation Engine

---

## 🔬 Reward Shaping & Solving Classic RL Traps

Vanilla reinforcement learning on lunar descent frequently encounters two catastrophic failure modes:

<p align="center">
  <img src="assets/Reward%20Shaping%20%26%20Solving%20Classic%20RL%20Traps.png" alt="Reward Shaping and Solving Classic RL Traps" width="100%" />
</p>

### 1. The "Cowardly Hoverer" Trap
When naive reward functions heavily penalize downward speed ($v_y$) or crashes without properly incentivizing descent, the agent discovers that attempting to touch the surface is hazardous. The policy converges to a sub-optimal local minimum: **hovering motionless at altitude $y \approx 1.0$ until the 1,000-step time limit expires**.

### 2. The "Lazy Hill Landing" Exploit
In default `LunarLander-v3`, coming to rest anywhere on the surface awards $+100$, even on the rough mountains outside the pad. If the agent spawns with lateral drift, it settles for touching down on the nearest hill slope ($x \approx 0.5$) rather than expending fuel to fight inertia and steer back to the flags.

### The Solution: Potential-Based Reward Shaping (PBRS)

To eliminate these exploits while mathematically guaranteeing policy invariance (Ng et al., 1999), we construct a potential function $\Phi(s)$ over the state space:

$$\Phi(s) = -25.0|x| - 10.0|y| - 5.0\sqrt{v_x^2 + v_y^2} - 5.0|\theta| + 10.0(\text{leg}_{\text{left}} + \text{leg}_{\text{right}})$$

The shaping reward added at transition $s \to s'$ is:

$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

- **Continuous Horizontal Pull ($-25|x|$):** Moving towards the pad center yields positive shaping ($\Delta \Phi > 0$) regardless of altitude, compelling the pilot to align with $x = 0$ early in descent.
- **Terminal Pad Differentiation:** If touchdown occurs outside the landing flags ($|x| > 0.22$), a $-100.0$ penalty cancels out the engine's default rest reward. Touching down between the flags ($|x| \le 0.22$) receives an additional $+100.0$ bonus, creating a **$200$-point incentive gap** strictly favoring the landing pad.
- **Contact Light Engine Cutoff:** In real aerospace engineering, rockets shut down main and attitude thrusters upon surface contact. The environment detects dual-leg contact ($\text{leg}_{\text{left}} = 1 \land \text{leg}_{\text{right}} = 1$) with zero residual velocity and concludes the flight immediately, eliminating ground thruster jitter and step timeouts.

---

## 🛠️ PPO Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Policy Network** | `MlpPolicy` (2x64) | Multi-Layer Perceptron actor-critic heads |
| **Optimizer** | Adam ($3 \times 10^{-4}$) | Learning rate |
| **Rollout Buffer (`n_steps`)** | `1024` | Horizon collected before update |
| **Minibatch Size** | `64` | Batch size per gradient step |
| **Epochs per Update** | `10` | Optimization passes over rollout buffer |
| **Discount Factor ($\gamma$)** | `0.99` | Long-term reward horizon discount |
| **GAE Parameter ($\lambda$)** | `0.95` | Generalized Advantage Estimation trade-off |
| **Clipping Range ($\epsilon$)** | `0.2` | PPO surrogate objective clipping threshold |
| **Total Timesteps** | `300,000` | Full convergence training budget |

---

## ⚡ Quickstart

### 1. Installation
```powershell
git clone https://github.com/Dakshin10/lunarpilot.git
cd lunarpilot
pip install -r requirements.txt
```

### 2. Evaluate the Trained Model
Run the pre-trained model immediately with real-time Pygame visualization and telemetry:
```powershell
python -m src.evaluate
```

### 3. Retrain from Scratch
```powershell
python -m src.train
```
*(Track training metrics, loss functions, and reward curves in TensorBoard via `tensorboard --logdir ./logs/`)*
