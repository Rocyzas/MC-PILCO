"""
PenSimPy environment wrapper for MC-PILCO.

State vector  : [X, S, DO2, pH, V, T, P, CO2outgas, O2, Wt, PAA, NH3, t_norm]
                (indices 0-12, STATE_DIM = 13)
Action vector : [Fs, Fpaa]  (indices 0-1, INPUT_DIM = 2)
                Foil, Fg, pressure, Fw are held at recipe defaults each step.

All states returned by rollout() are NORMALISED by STATE_SCALE so that each
dimension is roughly on a [0, 1] scale.  Every downstream component (policy,
GP, cost function) operates in this normalised space.

To hide penicillin until batch end (partial observability):
    Set P_OBSERVABLE_EVERY_STEP = False
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from pensimpy.peni_env_setup import PenSimEnv as _PenSimEnv
from pensimpy.data.constants import (
    DISCHARGE, FG, FG_DEFAULT_PROFILE,
    FOIL, FOIL_DEFAULT_PROFILE,
    FS, FS_DEFAULT_PROFILE,
    PAA, PAA_DEFAULT_PROFILE,
    PRES, PRESS_DEFAULT_PROFILE,
    WATER, WATER_DEFAULT_PROFILE,
)
from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.peni_env_setup import STEP_IN_MINUTES, NUM_STEPS

# ── Observability flags ────────────────────────────────────────────────────────
P_OBSERVABLE_EVERY_STEP = True

CONTROL_DISCHARGE = False
DISCHARGE_RATE    = 0.0   # fixed discharge [L/h]

# ── State indices ──────────────────────────────────────────────────────────────
IDX_X      = 0    # Biomass concentration          [g/L]
IDX_S      = 1    # Substrate concentration        [g/L]
IDX_DO2    = 2    # Dissolved oxygen               [mg/L]
IDX_PH     = 3    # pH                             [-]
IDX_V      = 4    # Vessel volume                  [L]
IDX_T      = 5    # Temperature                    [K]
IDX_P      = 6    # Penicillin concentration       [g/L]
IDX_CO2    = 7    # CO2 in off-gas                 [%]
IDX_O2     = 8    # O2 in off-gas                  [%]
IDX_WT     = 9    # Vessel weight                  [kg]
IDX_PAA    = 10   # PAA concentration              [g/L]
IDX_NH3    = 11   # NH3 concentration              [g/L]
IDX_TNORM  = 12   # Normalised batch time          [0..1]  (no noise)
STATE_DIM  = 13

# ── Action indices ─────────────────────────────────────────────────────────────
IDX_FS   = 0    # Sugar feed rate   [L/h]  — CONTROLLED by policy
IDX_FPAA = 1    # PAA feed rate     [L/h]  — CONTROLLED by policy
INPUT_DIM = 2   # Foil/Fg/pressure/Fw held at recipe defaults

# ── Physical action bounds ─────────────────────────────────────────────────────
# Recipe profile peaks: Fs ≈ 150 L/h, Fpaa ≈ 10 L/h.
# U_MAX adds headroom above those peaks.
U_MIN    = np.array([  0.0,   0.0], dtype=np.float64)   # [Fs, Fpaa]
U_MAX    = np.array([200.0,  12.0], dtype=np.float64)
U_CENTER = (U_MAX + U_MIN) / 2.0    # [100.0, 6.0]
U_HALF   = (U_MAX - U_MIN) / 2.0    # [100.0, 6.0]

# ── State normalisation ────────────────────────────────────────────────────────
STATE_SCALE = np.array(
    [5.0,      # X        [g/L]
     2.0,      # S        [g/L]
     20.0,     # DO2      [mg/L]
     8.0,      # pH       [-]
     6.5e4,    # V        [L]
     305.0,    # T        [K]
     10.0,     # P        [g/L]   recipe-driven reaches ~14 g/L
     0.1,      # CO2outgas [%]    init ≈ 0.038
     0.3,      # O2        [%]    init ≈ 0.20
     6.5e4,    # Wt        [kg]   init ≈ 62000
     2000.0,   # PAA       [g/L]  init ≈ 1400
     2000.0,   # NH3       [g/L]  init ≈ 1700
     1.0,      # t_norm    [-]    always in [0, 1]
    ],
    dtype=np.float64,
)

# Typical normalised initial state
INITIAL_STATE_NORM = np.array(
    [0.5   / STATE_SCALE[0],    # X
     1.0   / STATE_SCALE[1],    # S
     15.0  / STATE_SCALE[2],    # DO2
     6.5   / STATE_SCALE[3],    # pH
     5.8e4 / STATE_SCALE[4],    # V
     297.0 / STATE_SCALE[5],    # T
     0.0   / STATE_SCALE[6],    # P  (starts at 0)
     0.038 / STATE_SCALE[7],    # CO2outgas
     0.20  / STATE_SCALE[8],    # O2
     6.2e4 / STATE_SCALE[9],    # Wt
     1400. / STATE_SCALE[10],   # PAA
     1700. / STATE_SCALE[11],   # NH3
     0.0,                       # t_norm = 0 at batch start
    ],
    dtype=np.float64,
)

# Variance of the normalised initial state (batch-to-batch variation)
INITIAL_STATE_VAR_NORM = np.array(
    [(0.05  / STATE_SCALE[0])**2,   # X   ±0.05 g/L
     (0.10  / STATE_SCALE[1])**2,   # S   ±0.10 g/L
     (0.50  / STATE_SCALE[2])**2,   # DO2 ±0.5 mg/L
     (0.10  / STATE_SCALE[3])**2,   # pH  ±0.1
     (500.  / STATE_SCALE[4])**2,   # V   ±500 L
     (0.50  / STATE_SCALE[5])**2,   # T   ±0.5 K
     1e-8,                          # P   ~0 at start (tiny jitter for PD covariance)
     (0.001 / STATE_SCALE[7])**2,   # CO2outgas ±0.001 %
     (0.05  / STATE_SCALE[8])**2,   # O2  ±0.05 %
     (500.  / STATE_SCALE[9])**2,   # Wt  ±500 kg
     (50.   / STATE_SCALE[10])**2,  # PAA ±50 g/L
     (50.   / STATE_SCALE[11])**2,  # NH3 ±50 g/L
     1e-8,                          # t_norm always 0 at start (tiny jitter for PD covariance)
    ],
    dtype=np.float64,
)


# ── Fixed-action recipe interpolation ─────────────────────────────────────────
# Foil, Fg, pressure, Fw are not controlled by the policy; they follow the
# default IndPenSim recipe profiles, evaluated at the current batch time.

def _profile_to_arrays(profile_list):
    """Convert a pensimpy profile list-of-dicts to (times_h, values) numpy arrays."""
    times  = np.array([pt["time"]  for pt in profile_list], dtype=np.float64)
    values = np.array([pt["value"] for pt in profile_list], dtype=np.float64)
    return times, values

_FOIL_T,  _FOIL_V  = _profile_to_arrays(FOIL_DEFAULT_PROFILE)
_FG_T,    _FG_V    = _profile_to_arrays(FG_DEFAULT_PROFILE)
_PRESS_T, _PRESS_V = _profile_to_arrays(PRESS_DEFAULT_PROFILE)
_WATER_T, _WATER_V = _profile_to_arrays(WATER_DEFAULT_PROFILE)


def _recipe_interp(times_h, values, t_hours):
    """
    Linearly interpolate a recipe profile at time t_hours [h].
    np.interp clamps to first/last value outside the defined range.
    """
    return float(np.interp(t_hours, times_h, values))


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_default_recipe():
    """Return the standard IndPenSim recipe used by the PID sub-controllers
    inside the ODE.  Not used for RL control decisions."""
    _no_discharge = [{"time": 0, "value": 0}, {"time": 230, "value": 0}]
    return RecipeCombo(recipe_dict={
        FS:        Recipe(FS_DEFAULT_PROFILE,    FS),
        FOIL:      Recipe(FOIL_DEFAULT_PROFILE,  FOIL),
        FG:        Recipe(FG_DEFAULT_PROFILE,    FG),
        PRES:      Recipe(PRESS_DEFAULT_PROFILE, PRES),
        DISCHARGE: Recipe(_no_discharge,         DISCHARGE),
        WATER:     Recipe(WATER_DEFAULT_PROFILE, WATER),
        PAA:       Recipe(PAA_DEFAULT_PROFILE,   PAA),
    })


def _extract_state_raw(batch_data, k):
    """
    Read the raw (un-normalised) 12-variable state at the end of step k.
    t_norm is NOT included here — it is added deterministically in rollout().
    pH is stored internally as H+ concentration; converted and clamped to [0,14].
    """
    h_plus = float(batch_data.pH.y[k - 1])
    pH_val = np.clip(-math.log10(max(h_plus, 1e-14)), 0.0, 14.0)

    p_val = float(batch_data.P.y[k - 1])
    if not P_OBSERVABLE_EVERY_STEP:
        p_val = 0.0

    return np.array([
        float(batch_data.X.y[k - 1]),          # 0  X
        float(batch_data.S.y[k - 1]),          # 1  S
        float(batch_data.DO2.y[k - 1]),        # 2  DO2
        pH_val,                                 # 3  pH
        float(batch_data.V.y[k - 1]),          # 4  V
        float(batch_data.T.y[k - 1]),          # 5  T
        p_val,                                  # 6  P
        float(batch_data.CO2outgas.y[k - 1]),  # 7  CO2outgas
        float(batch_data.O2.y[k - 1]),         # 8  O2
        float(batch_data.Wt.y[k - 1]),         # 9  Wt
        float(batch_data.PAA.y[k - 1]),        # 10 PAA
        float(batch_data.NH3.y[k - 1]),        # 11 NH3
    ], dtype=np.float64)


def _initial_state_raw(env):
    """
    Reconstruct the t=0 raw 12-variable state from env.x0.
    t_norm is NOT included here.
    """
    h_plus = float(env.x0.pH)
    pH_val = np.clip(-math.log10(max(h_plus, 1e-14)), 0.0, 14.0)
    return np.array([
        float(env.x0.X),
        float(env.x0.S),
        float(env.x0.DO2),
        pH_val,
        float(env.x0.V),
        float(env.x0.T),
        float(env.x0.P),          # always 0.0
        float(env.x0.CO2outgas),
        float(env.x0.O2),
        float(env.x0.Wt),
        float(env.x0.PAA),
        float(env.x0.NH3),
    ], dtype=np.float64)


def _raw_to_physical(raw_u):
    """
    Map tanh-squashed policy output in (-1, 1) to physical [U_MIN, U_MAX].
    raw_u : np.ndarray [INPUT_DIM=2]  →  physical [Fs, Fpaa] in L/h
    """
    physical = U_CENTER + raw_u * U_HALF
    return np.clip(physical, U_MIN, U_MAX)


# ── Wrapper ────────────────────────────────────────────────────────────────────

class PenSimWrapper:
    """
    Wraps PenSimEnv to expose the rollout(s0, policy, T, dt, noise) interface
    expected by MC_PILCO.get_data_from_system().

    The policy controls only Fs and Fpaa.  Foil, Fg, pressure, and Fw are
    read from the default recipe profiles at each 12-min step.

    States returned are NORMALISED by STATE_SCALE (13 dimensions).

    Parameters
    ----------
    recipe_combo : RecipeCombo
        Background recipe for the PID sub-controllers inside the ODE.
        Use make_default_recipe().
    fast : bool
        Use fastodeint solver (True by default; set False if it crashes).
    seed_offset : int
        Added to rollout count to vary IC and disturbances across batches.
    """

    def __init__(self, recipe_combo, fast=True, seed_offset=0):
        self.recipe_combo   = recipe_combo
        self.fast           = fast
        self.seed_offset    = seed_offset
        self._rollout_count = 0

    def rollout(self, s0, policy, T, dt, noise):
        """
        Run one full batch under the given policy.

        Parameters
        ----------
        s0     : np.ndarray [STATE_DIM=13]
            Nominal initial state (normalised). Informational only.
        policy : callable(normalised_state, time_min) -> np.ndarray [INPUT_DIM=2]
            Returns tanh-squashed [raw_Fs, raw_Fpaa] in (-1, 1).
        T      : float   total duration [minutes]
        dt     : float   step size [minutes]  must equal STEP_IN_MINUTES (12)
        noise  : np.ndarray [STATE_DIM=13]  measurement noise std (normalised)
                 Set noise[IDX_TNORM] = 0.0 (t_norm is noiseless).

        Returns
        -------
        noisy_states     : np.ndarray [N+1, 13]  normalised, with noise
        inputs           : np.ndarray [N+1, 2]   raw tanh outputs stored for GP
        noiseless_states : np.ndarray [N+1, 13]  normalised, no noise
        """
        assert abs(dt - STEP_IN_MINUTES) < 1e-6, (
            f"dt must equal STEP_IN_MINUTES={STEP_IN_MINUTES:.1f} min, got {dt:.1f}."
        )
        num_steps = int(round(T / dt))

        env = _PenSimEnv(recipe_combo=self.recipe_combo, fast=self.fast)
        env.random_seed_ref = self.seed_offset + self._rollout_count
        env.yield_pre = 0
        self._rollout_count += 1

        _, batch_data = env.reset()

        noiseless = np.zeros((num_steps + 1, STATE_DIM), dtype=np.float64)
        inputs    = np.zeros((num_steps + 1, INPUT_DIM), dtype=np.float64)

        # t=0 state
        raw0              = _initial_state_raw(env)          # [12]
        noiseless[0, :12] = raw0 / STATE_SCALE[:12]
        noiseless[0, IDX_TNORM] = 0.0

        for k in range(1, num_steps + 1):
            t_min   = (k - 1) * dt          # minutes since batch start
            t_hours = t_min / 60.0          # hours since batch start

            # policy → raw tanh output → physical [Fs, Fpaa]
            raw_u    = np.asarray(policy(noiseless[k - 1], t_min)).flatten()
            physical = _raw_to_physical(raw_u)
            inputs[k - 1] = raw_u           # store tanh output for GP training

            Fs   = float(physical[IDX_FS])
            Fpaa = float(physical[IDX_FPAA])

            # fixed inputs from recipe interpolation
            Foil     = _recipe_interp(_FOIL_T,  _FOIL_V,  t_hours)
            Fg       = _recipe_interp(_FG_T,    _FG_V,    t_hours)
            pressure = _recipe_interp(_PRESS_T, _PRESS_V, t_hours)
            Fw       = _recipe_interp(_WATER_T, _WATER_V, t_hours)
            discharge = DISCHARGE_RATE

            _, batch_data, _, done = env.step(
                k, batch_data, Fs, Foil, Fg, pressure, discharge, Fw, Fpaa
            )

            raw_state = _extract_state_raw(batch_data, k)   # [12]

            # reveal P at the final step even if P_OBSERVABLE_EVERY_STEP is False
            if (not P_OBSERVABLE_EVERY_STEP) and (k == num_steps or done):
                raw_state[IDX_P] = float(batch_data.P.y[k - 1])

            noiseless[k, :12]   = raw_state / STATE_SCALE[:12]
            noiseless[k, IDX_TNORM] = k / num_steps          # deterministic

            if done:
                noiseless[k + 1:] = noiseless[k]
                inputs[k:]        = raw_u
                break

        # add Gaussian measurement noise (noise[IDX_TNORM] should be 0)
        noise_std = np.asarray(noise, dtype=np.float64).reshape(1, -1)
        noisy = noiseless + noise_std * np.random.randn(*noiseless.shape)

        return noisy, inputs, noiseless
