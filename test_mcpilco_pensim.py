"""
MC-PILCO for IndPenSim — base-case experiment.

State  : [X, S, DO2, pH, V, T, P, CO2outgas, O2, Wt, PAA, NH3, t_norm]
         (normalised, STATE_DIM = 13)
Actions: [Fs, Fpaa]  (tanh-squashed to (-1,1), INPUT_DIM = 2)
         Foil, Fg, pressure, Fw held at recipe defaults each step.
Reward : maximise terminal penicillin P(T) with PAA/NH3 constraint penalties

Run from project root:
    python test_mcpilco_pensim.py
"""

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# ── MC-PILCO imports ───────────────────────────────────────────────────────────
import policy_learning.MC_PILCO as MC_PILCO
import policy_learning.Policy as Policy
import policy_learning.Cost_function as Cost_function
import model_learning.Model_learning as Model_learning
import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import simulation_class.model as model   # only used for dummy f_sim trick

# ── Env imports ────────────────────────────────────────────────────────────────
from envs.pensim_env import (
    PenSimWrapper,
    make_default_recipe,
    STATE_DIM, INPUT_DIM,
    INITIAL_STATE_NORM, INITIAL_STATE_VAR_NORM,
    STEP_IN_MINUTES,
)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  Global settings
# ══════════════════════════════════════════════════════════════════════════════
dtype  = torch.float64
device = torch.device("cpu")

torch.manual_seed(0)
np.random.seed(0)

print("=" * 60)
print("MC-PILCO  ×  IndPenSim  —  base-case experiment")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Time and trial parameters
# ══════════════════════════════════════════════════════════════════════════════
T_sampling    = float(STEP_IN_MINUTES)   # 12.0 min — fixed by IndPenSim ODE
T_batch       = 230.0 * 60.0             # full batch duration [min]
T_exploration = T_batch
T_control     = T_batch

num_trials       = 5    # 1 random exploration + 4 learning trials
num_explorations = 1

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Dimensions derived from env
# ══════════════════════════════════════════════════════════════════════════════
state_dim    = STATE_DIM          # 13
input_dim    = INPUT_DIM          # 2
num_gp       = state_dim          # one GP per state dimension
gp_input_dim = state_dim + input_dim   # 15

print(f"\nstate_dim={state_dim}, input_dim={input_dim}, "
      f"num_gp={num_gp}, gp_input_dim={gp_input_dim}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  GP / model-learning configuration
# ══════════════════════════════════════════════════════════════════════════════
init_dict_RBF = {
    "active_dims":            np.arange(gp_input_dim),
    "lengthscales_init":      np.ones(gp_input_dim),
    "flg_train_lengthscales": True,
    "lambda_init":            np.ones(1),
    "flg_train_lambda":       True,
    "sigma_n_init":           0.01 * np.ones(1),
    "flg_train_sigma_n":      True,
    "sigma_n_num":            None,
    "dtype":                  dtype,
    "device":                 device,
}

model_learning_par = {
    "num_gp":           num_gp,
    "init_dict_list":   [init_dict_RBF] * num_gp,
    "flg_norm":         True,
    "dtype":            dtype,
    "device":           device,
}

model_optimization_opt_dict = {
    "f_optimizer":    "lambda p : torch.optim.Adam(p, lr=0.01)",
    "criterion":      Likelihood.Marginal_log_likelihood,
    "N_epoch":        1000,
    "N_epoch_print":  250,
}
# GP 12 (t_norm) is deterministic — freeze sigma_n so the optimizer can't
# collapse noise to zero and make K_X singular.
init_dict_RBF_tnorm = dict(init_dict_RBF)
init_dict_RBF_tnorm["sigma_n_init"]      = 1e-2 * np.ones(1)
init_dict_RBF_tnorm["flg_train_sigma_n"] = False

model_learning_par["init_dict_list"] = [init_dict_RBF] * (num_gp - 1) + [init_dict_RBF_tnorm]

model_optimization_opt_list = [model_optimization_opt_dict] * num_gp

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Exploration policy  (uniform random)
# ══════════════════════════════════════════════════════════════════════════════
f_rand_exploration_policy = Policy.Random_exploration
rand_exploration_policy_par = {
    "state_dim":  state_dim,
    "input_dim":  input_dim,
    "u_max":      np.ones(input_dim),
    "dtype":      dtype,
    "device":     device,
}

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Control policy  (sum of Gaussians)
# ══════════════════════════════════════════════════════════════════════════════
num_basis    = 200
centers_init = np.random.rand(num_basis, state_dim)   # [200, 13]

f_control_policy = Policy.Sum_of_gaussians
control_policy_par = {
    "state_dim":              state_dim,
    "input_dim":              input_dim,
    "num_basis":              num_basis,
    "centers_init":           centers_init,
    "lengthscales_init":      np.ones(state_dim),
    "flg_train_lengthscales": True,
    "flg_train_centers":      True,
    "weight_init":            (np.random.rand(input_dim, num_basis) - 0.5) * 0.1,
    "flg_squash":             True,
    "u_max":                  np.ones(input_dim),
    "flg_drop":               True,
    "dtype":                  dtype,
    "device":                 device,
}

policy_reinit_dict = {
    "lenghtscales_par": np.ones(state_dim),
    "centers_par":      np.random.rand(num_basis, state_dim),
    "weight_par":       (np.random.rand(input_dim, num_basis) - 0.5) * 0.1,
}

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Policy optimisation
# ══════════════════════════════════════════════════════════════════════════════
policy_optimization_dict = {
    "num_particles":       100,
    "opt_steps_list":      [2000] * num_trials,
    "lr_list":             [0.01] * num_trials,
    "f_optimizer":         "lambda p, lr : torch.optim.Adam(p, lr)",
    "num_step_print":      200,
    "p_dropout_list":      [0.25] * num_trials,
    "p_drop_reduction":    0.125,
    "alpha_diff_cost":     0.99,
    "min_diff_cost":       0.01,
    "num_min_diff_cost":   200,
    "min_step":            200,
    "lr_min":              0.001,
    "policy_reinit_dict":  policy_reinit_dict,
}

# ══════════════════════════════════════════════════════════════════════════════
# 7.  Cost function
# ══════════════════════════════════════════════════════════════════════════════
# Terminal P reward + PAA/NH3 constraint penalties + action regularisation.
# PAA bounds: 600–1800 (normalised: 0.30–0.90 with STATE_SCALE[10]=2000)
# NH3 lower:  300      (normalised: 0.15         with STATE_SCALE[11]=2000)
f_cost_function  = Cost_function.PenSimCost_v2
cost_function_par = {
    "p_idx":       6,
    "paa_idx":     10,
    "nh3_idx":     11,
    "paa_lo_norm": 0.30,
    "paa_hi_norm": 0.90,
    "nh3_lo_norm": 0.15,
    "w_P":         1.0,
    "lambda_u":    1e-3,
    "lambda_du":   1e-3,
}

# ══════════════════════════════════════════════════════════════════════════════
# 8.  Measurement noise (in normalised units)
# ══════════════════════════════════════════════════════════════════════════════
std_meas_noise = np.array([
    0.01,   # X         (±0.05 g/L   / 5.0)
    0.02,   # S         (±0.04 g/L   / 2.0)
    0.02,   # DO2       (±0.4 mg/L   / 20)
    0.01,   # pH        (±0.08       / 8.0)
    0.005,  # V         (±325 L      / 65000)
    0.002,  # T         (±0.6 K      / 305)
    0.01,   # P         (±0.1 g/L    / 10)
    0.02,   # CO2outgas
    0.02,   # O2
    0.005,  # Wt
    0.02,   # PAA
    0.02,   # NH3
    0.0,    # t_norm    — noiseless by construction
])

# ══════════════════════════════════════════════════════════════════════════════
# 9.  Assemble MC-PILCO
# ══════════════════════════════════════════════════════════════════════════════
print("\n---- Initialising MC-PILCO ----")

log_path = os.path.join(_ROOT, "results", "pensim")
os.makedirs(log_path, exist_ok=True)

MC_PILCO_init_dict = {
    "T_sampling":                   T_sampling,
    "state_dim":                    state_dim,
    "input_dim":                    input_dim,
    "f_sim":                        lambda x, t, u: np.zeros_like(x),
    "f_model_learning":             Model_learning.Model_learning_RBF,
    "model_learning_par":           model_learning_par,
    "f_rand_exploration_policy":    f_rand_exploration_policy,
    "rand_exploration_policy_par":  rand_exploration_policy_par,
    "f_control_policy":             f_control_policy,
    "control_policy_par":           control_policy_par,
    "f_cost_function":              f_cost_function,
    "cost_function_par":            cost_function_par,
    "std_meas_noise":               std_meas_noise,
    "log_path":                     log_path,
    "dtype":                        dtype,
    "device":                       device,
}

mc_pilco = MC_PILCO.MC_PILCO(**MC_PILCO_init_dict)

mc_pilco.system = PenSimWrapper(
    recipe_combo=make_default_recipe(),
    fast=True,
    seed_offset=0,
)
print("PenSimWrapper attached.")

# ══════════════════════════════════════════════════════════════════════════════
# 10.  Run
# ══════════════════════════════════════════════════════════════════════════════
reinforce_param_dict = {
    "initial_state":     INITIAL_STATE_NORM,
    "initial_state_var": INITIAL_STATE_VAR_NORM,
    "T_exploration":     T_exploration,
    "T_control":         T_control,
    "num_trials":        num_trials,
    "num_explorations":  num_explorations,
    "model_optimization_opt_list": model_optimization_opt_list,
    "policy_optimization_dict":    policy_optimization_dict,
}

print("\n---- Starting reinforce() ----\n")
cost_list, particles_states, particles_inputs = mc_pilco.reinforce(**reinforce_param_dict)

print("\n---- Done ----")
print(f"Cost per trial: {cost_list}")

# ── Plot results ───────────────────────────────────────────────────────────────
import subprocess
subprocess.run(
    [sys.executable, os.path.join(_ROOT, "plot_pensim_results.py"),
     "--log",  os.path.join(_ROOT, "results", "pensim", "log.pkl"),
     "--out",  os.path.join(_ROOT, "results", "pensim")],
    check=False,
)
