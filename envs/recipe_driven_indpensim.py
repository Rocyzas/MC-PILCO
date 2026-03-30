"""
Recipe-Driven Baseline for IndPenSim

Runs the default factory recipe across multiple seeds (batches) with no learning
or optimisation.

Run from project root:
    python recipe_driven_indpensim.py
    python recipe_driven_indpensim.py --n_seeds 20
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make pensimpy importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from pensimpy.data.constants import (
    DISCHARGE, DISCHARGE_DEFAULT_PROFILE,
    FG,   FG_DEFAULT_PROFILE, # Aeration Flow rate
    FOIL, FOIL_DEFAULT_PROFILE, # Soybean Flow oil rate
    FS,   FS_DEFAULT_PROFILE, # Sugar Flow rate
    PAA,  PAA_DEFAULT_PROFILE, # Phenylacetic acid concentration - Acid fed into broth IMPORTANT
    PRES, PRESS_DEFAULT_PROFILE, # pressure
    WATER, WATER_DEFAULT_PROFILE, # Water flow rate
)
from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.peni_env_setup import (
    MINUTES_PER_HOUR,
    NUM_STEPS, # 230 * 60 (BATCH_LENGTH_IN_MINUTES) / 12 (step in minutes)
    STEP_IN_HOURS,
    STEP_IN_MINUTES,
    PenSimEnv as _PenSimEnv,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser("Recipe-driven baseline")
p.add_argument("--n_seeds", type=int, default=10, help="Number of batch seeds to run")
p.add_argument("--base_seed", type=int, default=1, help="Starting seed")
args = p.parse_args()

N_SEEDS   = args.n_seeds
BASE_SEED = args.base_seed

# ---------------------------------------------------------------------------
# Build default recipe (used for the entire batch — no agent control)
# ---------------------------------------------------------------------------
def make_default_recipe():
    return RecipeCombo(recipe_dict={
        FS:        Recipe(FS_DEFAULT_PROFILE,        FS),
        FOIL:      Recipe(FOIL_DEFAULT_PROFILE,      FOIL),
        FG:        Recipe(FG_DEFAULT_PROFILE,        FG),
        PRES:      Recipe(PRESS_DEFAULT_PROFILE,     PRES),
        DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
        WATER:     Recipe(WATER_DEFAULT_PROFILE,     WATER),
        PAA:       Recipe(PAA_DEFAULT_PROFILE,       PAA),
    })

# ---------------------------------------------------------------------------
# Run one batch with the default recipe, return full P trajectory
# ---------------------------------------------------------------------------
def run_recipe_batch(seed: int):
    """
    Run a full batch under default recipe control.
    Returns:
        time_hours  : array of simulation times [h]
        pen_traj    : penicillin concentration trajectory [g/L]
        max_pen     : peak penicillin concentration [g/L]
        final_pen   : penicillin concentration at batch end [g/L]
        n_steps     : number of steps completed
    """
    recipe = make_default_recipe()
    env = _PenSimEnv(recipe_combo=recipe, fast=True)
    env.random_seed_ref = seed
    env.yield_pre = 0

    obs, batch_data = env.reset()

    time_hours = []
    pen_traj   = []

    for k in range(1, NUM_STEPS + 1):
        t_hr = k * STEP_IN_MINUTES / MINUTES_PER_HOUR
        v    = recipe.get_values_dict_at(time=t_hr)

        obs, batch_data, reward, done = env.step(
            k, batch_data,
            v["Fs"], v["Foil"], v["Fg"],
            v["pressure"], v["discharge"], v["Fw"], v["Fpaa"],
        )

        # Read penicillin from batch_data
        p_val = float(batch_data.P.y[k - 1])
        if math.isnan(p_val) or p_val < 0.0:
            p_val = pen_traj[-1] if pen_traj else 0.0

        time_hours.append(t_hr)
        pen_traj.append(p_val)

        if done:
            break

    # Penicilin trajectory and time arrays
    pen_traj   = np.array(pen_traj)
    time_hours = np.array(time_hours)

    # Find last valid (non-NaN, positive) penicillin value
    valid = [v for v in pen_traj if v > 0 and not math.isnan(v)]
    max_pen   = float(np.max(pen_traj))
    final_pen = valid[-1] if valid else 0.0

    return time_hours, pen_traj, max_pen, final_pen, k

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print("=" * 60)
print("Recipe-Driven Baseline")
print(f"Seeds {BASE_SEED} to {BASE_SEED + N_SEEDS - 1}  ({N_SEEDS} batches)")
print("=" * 60)
print(f"\n{'Seed':<8} {'Steps':>6} {'max_P [g/L]':>12} {'final_P [g/L]':>14}")
print("-" * 44)

all_trajs    = []
all_times    = []
all_max_pen  = []
all_final_pen = []

for i in range(N_SEEDS):
    seed = BASE_SEED + i
    t, pen, max_p, final_p, steps = run_recipe_batch(seed)
    all_trajs.append(pen)
    all_times.append(t)
    all_max_pen.append(max_p)
    all_final_pen.append(final_p)
    print(f"{seed:<8} {steps:>6} {max_p:>12.3f} {final_p:>14.3f}")

all_max_pen   = np.array(all_max_pen)
all_final_pen = np.array(all_final_pen)

print("-" * 44)
print(f"\n{'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 44)
print(f"{'max_P [g/L]':<20} {all_max_pen.mean():>8.3f} {all_max_pen.std():>8.3f} "
      f"{all_max_pen.min():>8.3f} {all_max_pen.max():>8.3f}")
print(f"{'final_P [g/L]':<20} {all_final_pen.mean():>8.3f} {all_final_pen.std():>8.3f} "
      f"{all_final_pen.min():>8.3f} {all_final_pen.max():>8.3f}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
results_dir = os.path.join(_HERE, "results", "recipe_baseline")
os.makedirs(results_dir, exist_ok=True)

colors = plt.cm.tab10(np.linspace(0, 0.9, N_SEEDS))

# --- Plot 1: Penicillin trajectories ---
fig, ax = plt.subplots(figsize=(10, 5))
for i, (t, pen) in enumerate(zip(all_times, all_trajs)):
    ax.plot(t, pen, color=colors[i], alpha=0.8, linewidth=1.2,
            label=f"seed {BASE_SEED + i}")
ax.set_xlabel("Time [h]")
ax.set_ylabel("Penicillin concentration [g/L]")
ax.set_title(f"Recipe-Driven Baseline — Penicillin Trajectories ({N_SEEDS} batches)")
ax.legend(fontsize=7, ncol=max(1, N_SEEDS // 5), loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
traj_path = os.path.join(results_dir, "pen_trajectories.png")
fig.savefig(traj_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nTrajectory plot → {traj_path}")

# --- Plot 2: Yield distribution (bar chart with mean +- std) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart per seed
seed_labels = [str(BASE_SEED + i) for i in range(N_SEEDS)]
x = np.arange(N_SEEDS)

axes[0].bar(x, all_max_pen, color=colors, alpha=0.85, label="max_P per batch")
axes[0].axhline(all_max_pen.mean(), color="red", linewidth=1.5,
                linestyle="--", label=f"mean = {all_max_pen.mean():.2f} g/L")
axes[0].fill_between([-0.5, N_SEEDS - 0.5],
                     all_max_pen.mean() - all_max_pen.std(),
                     all_max_pen.mean() + all_max_pen.std(),
                     alpha=0.15, color="red", label=f"+-1 std = {all_max_pen.std():.2f} g/L")
axes[0].set_xticks(x)
axes[0].set_xticklabels(seed_labels, fontsize=8)
axes[0].set_xlabel("Batch seed")
axes[0].set_ylabel("Max penicillin [g/L]")
axes[0].set_title("Peak Penicillin per Batch")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, axis="y")

# Histogram of max_P values
axes[1].hist(all_max_pen, bins=min(8, N_SEEDS), color="steelblue", edgecolor="white",
             alpha=0.85)
axes[1].axvline(all_max_pen.mean(), color="red", linewidth=1.5,
                linestyle="--", label=f"mean = {all_max_pen.mean():.2f} g/L")
axes[1].axvline(all_max_pen.mean() - all_max_pen.std(), color="red",
                linewidth=1.0, linestyle=":", alpha=0.7)
axes[1].axvline(all_max_pen.mean() + all_max_pen.std(), color="red",
                linewidth=1.0, linestyle=":", alpha=0.7,
                label=f"+-1 std = {all_max_pen.std():.2f} g/L")
axes[1].set_xlabel("Max penicillin [g/L]")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of Peak Penicillin")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis="y")

fig.suptitle(f"Recipe-Driven Baseline ({N_SEEDS} batches, seeds {BASE_SEED}–{BASE_SEED+N_SEEDS-1})",
             fontsize=12, fontweight="bold")
fig.tight_layout()
dist_path = os.path.join(results_dir, "yield_distribution.png")
fig.savefig(dist_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Distribution plot  → {dist_path}")
print("\nDone.")
