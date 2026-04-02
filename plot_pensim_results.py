"""
Plot MC-PILCO × IndPenSim results from a saved log.pkl.

Usage (from project root):
    python plot_pensim_results.py
    python plot_pensim_results.py --log results/pensim/log.pkl --out results/pensim/

Produces four figure files:
    1_penicillin.png          — P trajectory for every real rollout
    2_states.png              — all 7 state variables across rollouts
    3_actions.png             — all 6 control inputs across rollouts
    4_policy_cost.png         — policy optimisation convergence per trial
"""

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from envs.pensim_env import STATE_SCALE, INPUT_DIM, U_MIN, U_MAX, U_CENTER, U_HALF, STEP_IN_MINUTES

# ── Labels ─────────────────────────────────────────────────────────────────────
STATE_NAMES = [
    "Biomass X", "Substrate S", "Dissolved O₂", "pH", "Volume V", "Temp T", "Penicillin P",
    "CO₂ off-gas", "O₂ off-gas", "Vessel Wt", "PAA conc.", "NH₃ conc.", "Time (norm)",
]
STATE_UNITS = [
    "g/L", "g/L", "mg/L", "–", "L", "K", "g/L",
    "%", "%", "kg", "g/L", "g/L", "–",
]
STATE_IDX_P = 6

ACTION_NAMES = ["Fs (sugar feed)", "Fpaa (PAA feed)"]
ACTION_UNITS = ["L/h",             "L/h"]

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--log", default=os.path.join(_ROOT, "results", "pensim", "log.pkl"),
                    help="Path to log.pkl")
parser.add_argument("--out", default=os.path.join(_ROOT, "results", "pensim"),
                    help="Directory to save figures")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
if not os.path.exists(args.log):
    print(f"Log file not found: {args.log}")
    print("Run test_mcpilco_pensim.py first, or pass --log <path>.")
    sys.exit(1)

with open(args.log, "rb") as f:
    log = pickle.load(f)

# Real rollout histories  (list of np.arrays, each [N+1, state_dim] normalised)
state_hist   = log.get("state_samples_history",    [])
input_hist   = log.get("input_samples_history",    [])
cost_list    = log.get("cost_trial_list",           [])   # per-trial policy opt costs
std_list     = log.get("std_cost_trial_list",       [])

# Particle rollouts from the planned trajectory (list of tensors [T, N_par, state_dim])
particles_states = log.get("particles_states_list", [])

n_rollouts = len(state_hist)
n_trials   = len(cost_list)
print(f"Loaded: {n_rollouts} rollout(s), {n_trials} trial(s) with policy cost data.")

if n_rollouts == 0:
    print("No rollout data yet — run at least one exploration trial.")
    sys.exit(0)

# ── Helpers ────────────────────────────────────────────────────────────────────
COLORS = plt.cm.tab10(np.linspace(0, 0.9, max(n_rollouts, 1)))

def rollout_label(i):
    return "Exploration" if i == 0 else f"Trial {i}"

def time_hours(n_steps):
    """Return time axis in hours for a trajectory of length n_steps."""
    return np.arange(n_steps) * STEP_IN_MINUTES / 60.0

def denorm_states(arr_norm):
    """Multiply normalised state array by STATE_SCALE to recover physical units."""
    return arr_norm * STATE_SCALE

def to_numpy(x):
    """Convert tensor or ndarray to numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Penicillin concentration across rollouts
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))

for i, s_norm in enumerate(state_hist):
    s = denorm_states(to_numpy(s_norm))
    t = time_hours(s.shape[0])
    ax.plot(t, s[:, STATE_IDX_P], color=COLORS[i], linewidth=1.4,
            label=rollout_label(i))

ax.set_xlabel("Time [h]")
ax.set_ylabel("Penicillin P [g/L]")
ax.set_title("Penicillin Concentration — Real Rollouts")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0)
ax.set_ylim(bottom=0)
fig.tight_layout()
path = os.path.join(args.out, "1_penicillin.png")
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — All state variables across rollouts
# ══════════════════════════════════════════════════════════════════════════════
n_states = len(STATE_NAMES)
ncols = 2
nrows = (n_states + 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 2.8))
axes = axes.flatten()

for j in range(n_states):
    ax = axes[j]
    for i, s_norm in enumerate(state_hist):
        s = denorm_states(to_numpy(s_norm))
        t = time_hours(s.shape[0])
        ax.plot(t, s[:, j], color=COLORS[i], linewidth=1.2,
                label=rollout_label(i) if j == 0 else None)
    ax.set_title(f"{STATE_NAMES[j]} [{STATE_UNITS[j]}]", fontsize=9)
    ax.set_xlabel("Time [h]", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0)

# Hide unused subplot if odd number of state variables
if n_states % ncols != 0:
    axes[-1].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", fontsize=9, ncol=min(n_rollouts, 4))
fig.suptitle("State Variables — Real Rollouts", fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0.05, 1, 1])
path = os.path.join(args.out, "2_states.png")
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Control action profiles across rollouts
# ══════════════════════════════════════════════════════════════════════════════
if len(input_hist) > 0:
    n_inputs = INPUT_DIM
    ncols_a  = 2
    nrows_a  = (n_inputs + 1) // ncols_a
    fig, axes = plt.subplots(nrows_a, ncols_a, figsize=(12, nrows_a * 2.8))
    axes = axes.flatten()

    for j in range(n_inputs):
        ax = axes[j]
        for i, u_arr in enumerate(input_hist):
            u_raw = to_numpy(u_arr)[:-1, :]   # raw tanh ∈ (−1,1), drop padded last row
            u = u_raw * U_HALF[None, :] + U_CENTER[None, :]  # convert to physical units
            t = time_hours(u.shape[0])
            ax.plot(t, u[:, j], color=COLORS[i], linewidth=1.0,
                    label=rollout_label(i) if j == 0 else None)
        # mark physical bounds as horizontal dashed lines
        ax.axhline(U_MIN[j], color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axhline(U_MAX[j], color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{ACTION_NAMES[j]} [{ACTION_UNITS[j]}]", fontsize=9)
        ax.set_xlabel("Time [h]", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0)

    if n_inputs % ncols_a != 0:
        axes[-1].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9, ncol=min(n_rollouts, 4))
    fig.suptitle("Control Actions — Real Rollouts", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(args.out, "3_actions.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Policy optimisation cost curves
# ══════════════════════════════════════════════════════════════════════════════
if n_trials > 0:
    trial_colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_trials))
    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (costs, stds) in enumerate(zip(cost_list, std_list)):
        c = to_numpy(costs) if not isinstance(costs, np.ndarray) else costs
        s = to_numpy(stds)  if not isinstance(stds,  np.ndarray) else stds
        steps = np.arange(len(c))
        ax.plot(steps, c, color=trial_colors[i], linewidth=1.3, label=f"Trial {i + 1}")
        ax.fill_between(steps, c - s, c + s, color=trial_colors[i], alpha=0.15)

    ax.set_xlabel("Optimisation step")
    ax.set_ylabel("Expected cost")
    ax.set_title("Policy Optimisation Convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.out, "4_policy_cost.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Planned particle trajectories (last trial, P only)
# ══════════════════════════════════════════════════════════════════════════════
if len(particles_states) > 0:
    last_particles = to_numpy(particles_states[-1])  # [T, N_particles, state_dim]
    T, N_par, _ = last_particles.shape
    t = time_hours(T)

    P_particles_norm = last_particles[:, :, STATE_IDX_P]           # [T, N_par] normalised
    P_particles      = P_particles_norm * STATE_SCALE[STATE_IDX_P]  # physical [g/L]
    P_mean           = P_particles.mean(axis=1)
    P_std            = P_particles.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, P_mean, color="steelblue", linewidth=1.6, label="Particle mean")
    ax.fill_between(t, P_mean - P_std, P_mean + P_std,
                    color="steelblue", alpha=0.25, label="±1 std (particles)")
    ax.fill_between(t, P_mean - 2*P_std, P_mean + 2*P_std,
                    color="steelblue", alpha=0.10, label="±2 std (particles)")

    # overlay real rollout if available
    if len(state_hist) > 1:
        s_real = denorm_states(to_numpy(state_hist[-1]))
        t_real = time_hours(s_real.shape[0])
        ax.plot(t_real, s_real[:, STATE_IDX_P], color="tomato",
                linewidth=1.3, linestyle="--", label="Real rollout (last trial)")

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Penicillin P [g/L]")
    ax.set_title(f"Planned Trajectory — Last Trial (N={N_par} particles)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = os.path.join(args.out, "5_planned_trajectory.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

print("\nDone.")
