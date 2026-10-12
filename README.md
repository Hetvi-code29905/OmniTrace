# OmniTrace

> **Hierarchical Autonomous Navigation: Coupling Classical Global Route Planning with Local Reinforcement Learning to Eliminate the Freezing Robot Problem.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pygame](https://img.shields.io/badge/UI-Pygame%202.6-green.svg)](https://www.pygame.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Hierarchical%20RL-purple.svg)](#core-architecture)

---

## Overview & Problem Statement

In real-world mobile autonomy, navigation systems face two persistent failure modes when dynamic obstacles appear:
1. **The Computational Bottleneck of Global Replanning:** In large environments, recalculating full geometric routes (e.g., standard Dijkstra or A\*) every time a moving hazard enters the path is computationally expensive and introduces dangerous latency.
2. **The Freezing Robot Problem:** When corridors are narrow or temporarily blocked, geometric planners conclude that no safe path exists and bring the mobile platform to a complete halt—even if the obstacle is moving and clearing the way shortly.

**OmniTrace** demonstrates a **two-tier hierarchical navigation architecture**:
- **Macro Layer (Global Dijkstra):** Solves the global topological shortest-path problem on the static environment.
- **Micro Layer (Local Q-Learning Reflex Policy):** Performs instantaneous, $O(1)$ constant-time tactical arbitration directly from raw 36-ray LiDAR sector clearances to nudge past obstacles or yield without thrashing global route memory.
- **Unfreezing Watchdog:** Monitors tactical yield duration to trigger strategic detours if a corridor remains physically impassable.

---

## Core Architecture

```
                                 [ Destination Goal (x, y) ]
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │ Global Dijkstra Plan │  <── (Only triggered on startup
                                   │ (Topological Route)  │       or persistent blockage)
                                   └──────────┬───────────┘
                                              │ Global Waypoints
                                              ▼
┌──────────────────────┐           ┌──────────────────────┐
│  36-Ray 2D LiDAR     ├──────────►│ State Discretizer    │
│  Perception Scan     │ Clearances│ (Front, Left, Right) │
└──────────────────────┘ (F, L, R) └──────────┬───────────┘
                                              │ Local State s ∈ {0..7}
                                              ▼
                                   ┌──────────────────────┐
                                   │ Local Q-Learning     │  <── O(1) Constant-Time Lookup
                                   │ Policy (Q-Table)     │      (CRUISE, YIELD, BYPASS_L, BYPASS_R)
                                   └──────────┬───────────┘
                                              │ Tactical Action
                                              ▼
                                   ┌──────────────────────┐
                                   │ Velocity Command     │
                                   │ Arbitrator (v, ω)    │
                                   └──────────┬───────────┘
                                              │
                                              ▼
                                   [ Differential-Drive AMR ]
```

### The 8-State Discretized Local Policy
Rather than feeding high-dimensional LiDAR vectors into slow neural networks, OmniTrace partitions the continuous laser raycast into 3 critical geometric corridors aligned with the robot's upcoming path vector:
- **Front Sector ($\pm 22^\circ$):** Safety corridor ($d_{thresh} = 0.68\,\text{m}$)
- **Left Sector ($0^\circ \text{ to } +95^\circ$):** Left clearance ($d_{thresh} = 0.52\,\text{m}$)
- **Right Sector ($-95^\circ \text{ to } 0^\circ$):** Right clearance ($d_{thresh} = 0.52\,\text{m}$)

This forms an 8-state discrete representation:
$$\text{State } s = (B_{\text{front}} \ll 2) \mid (B_{\text{left}} \ll 1) \mid B_{\text{right}} \in [0, 7]$$

### Action Set & Tactical Arbitration
- `CRUISE`: Follow the pure-pursuit waypoint tracking trajectory.
- `YIELD_WAIT`: Decelerate to $(v=0, \omega=0)$ to allow moving hazards to cross.
- `BYPASS_LEFT`: Inject a positive angular velocity bias to slip around right-side obstructions.
- `BYPASS_RIGHT`: Inject a negative angular velocity bias to slip around left-side obstructions.

---

## Interactive Simulation Environments

OmniTrace features **3 dedicated simulation environments** switchable in real-time:

### `[Mode 1] GLOBAL GOD`
- Full-knowledge static grid navigation.
- Continuous unicycle kinematics integrated at 30 FPS.
- Dead-reckoning odometry with cumulative Gaussian noise and wheel slip.
- AMCL Ground-Truth Sync (`[C]`).
- Sudden moving obstacle demonstration (`[D]`) showcasing blind collisions when operating without active perception.

### `[Mode 2] LIDAR FOG`
- Exploration under total perceptual fog of war.
- 360-degree, 36-ray raymarching LiDAR sensor (range: 6.0m).
- 2D Occupancy Grid mapping (`Unknown`, `Free`, `Occupied`).
- Dynamic real-time obstacle avoidance and optimistic free-space path replanning.

### `[Mode 3] HYBRID RL`
- Complete hierarchical autonomy stack.
- Real-time Q-Table inspector and LiDAR sector clearance telemetry in the HUD.
- Autonomous moving hazard interaction: AMR detects oncoming dynamic objects, yields safely, or bypasses them.
- Visual crash impact shockwave and 2-second hold-and-inspect collision recovery.
- Freezing Robot Watchdog: If blocked longer than 40 ticks (~1.3s), automatically triggers an unfreezing global detour.

---

## Why RL Over Full-Map Graph Replanning ($D^*$ Lite)?

| Feature | Graph Replanning (Dijkstra / $D^*$ Lite) | OmniTrace Hybrid RL |
| :--- | :--- | :--- |
| **Response Latency** | Variable $O(V \log V)$ to $O(E)$ graph expansion | **Guaranteed $O(1)$ microsecond table lookup** |
| **Behavior in Narrow Aisle** | Oscillates or aborts if the path is temporarily blocked | **Tactically yields and waits for obstacle to pass** |
| **Global Path Churn** | Re-routes the whole journey around transient obstacles | **Keeps global route intact; executes local micro-tactics** |
| **Freezing Problem** | Freezes when all immediate paths are blocked | **Unfreezing watchdog actively breaks deadlocks** |

---

## Quick Start

### 1. Prerequisites & Environment Setup
Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/<your-username>/OmniTrace.git
cd OmniTrace

# Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e .
```

### 2. Launch the Interactive Simulator
```bash
python -m atlas_robotics.main
```

---

## Controls Cheat Sheet

| Key / Action | Function |
| :--- | :--- |
| **`1`, `2`, `3`** / **Top Tabs** | Switch between Modes: `GLOBAL GOD`, `LIDAR FOG`, `HYBRID RL` |
| **`Space`** | Toggle Autonomous Mission Execution (Run / Pause) |
| **`D`** | Toggle Dynamic Moving Hazard in the environment |
| **`Left-Click + Drag`** | Paint static obstacles in real-time |
| **`Right-Click + Drag`** | Erase obstacles / Relocate Goal position |
| **`Shift + Left-Click`** | Teleport AMR to clicked location |
| **`C`** | Sync Odometry with Ground-Truth pose (AMCL calibration) |
| **`R`** | Reset active stage to default factory state |
| **`W, A, S, D` / Arrows** | Manual teleoperation (when autonomous drive is paused) |
| **`ESC`** | Exit simulator |

---

## Project Structure

```
OmniTrace/
├── src/
│   └── atlas_robotics/
│       ├── main.py                  # Pygame visualizer, topbar tabs, telemetry HUD
│       ├── core/
│       │   ├── grid.py              # 2D GridWorld and obstacle representation
│       │   └── kinematics.py        # Differential-drive kinematic equations
│       ├── planning/
│       │   └── dijkstra.py          # Classical Dijkstra global route planner
│       ├── sensors/
│       │   ├── lidar.py             # 360-degree 36-ray laser raycaster
│       │   └── mapping.py           # Probabilistic 2D Occupancy Grid map
│       ├── localization/
│       │   └── odometry.py          # Dead-reckoning estimator with wheel slip noise
│       ├── control/
│       │   └── waypoint_follower.py # Continuous unicycle waypoint pursuit controller
│       └── learning/
│           ├── local_rl_policy.py   # 8-state tabular Q-learning reflex policy
│           └── hybrid_navigator.py  # Hierarchical arbitration coordinator
├── pyproject.toml                   # Project metadata and build configuration
└── README.md                        # Documentation
```

---

## License
MIT License. Open-source for academic research, education, and autonomous systems experimentation.
