# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


class Expected_cost(torch.nn.modules.loss._Loss):
    """
    Expected cost class. The cost is computed through a torch function defined in the initialization
    """

    def __init__(self, cost_function):
        """Initialize the object"""
        super(Expected_cost, self).__init__()
        self.cost_function = cost_function

    def forward(self, states_sequence, inputs_sequence, trial_index=None):
        """Computes the global cost applying self.cost_function.
        States_sequence.shape: [num_instants, num_particles, state_dim]
        inputs_sequence.shape: [num_instants, num_particles, input_dim]
        """

        # Returns the sum of the expected costs
        costs = self.cost_function(states_sequence, inputs_sequence, trial_index)
        mean_costs = torch.mean(costs, 1)  # average cost at each time step over particles ...
        std_costs = torch.std(costs.detach(), 1)  # ... and corresponding std

        return torch.sum(mean_costs), torch.sum(std_costs)


class Expected_distance(Expected_cost):
    """
    Cost function given by the sum of the expected distances from target state
    """

    def __init__(self, target_state, lengthscales, active_dims):
        # get the distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: distance_from_target(
            x, u, trial_index, target_state=target_state, lengthscales=lengthscales, active_dims=active_dims
        )
        # initit the superclass with the lambda function
        super(Expected_distance, self).__init__(f_cost)


def distance_from_target(states_sequence, inputs_sequence, trial_index, target_state, lengthscales, active_dims):
    # normalize states and targets (consider only used states)
    norm_states = states_sequence[:, :, active_dims] / lengthscales
    norm_target = target_state / lengthscales

    # get the square distance
    dist = torch.sum(norm_states**2, dim=2, keepdim=True)
    dist = dist + torch.sum(norm_target**2, dim=1, keepdim=True).transpose(0, 1)
    dist -= 2 * torch.matmul(norm_states, norm_target.transpose(dim0=0, dim1=1))
    # return the cost
    return dist


class Expected_saturated_distance(Expected_cost):
    """
    Cost function given by the sum of the expected saturated distances from target state
    """

    def __init__(self, target_state, lengthscales, active_dims):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: saturated_distance_from_target(
            x, u, trial_index, target_state=target_state, lengthscales=lengthscales, active_dims=active_dims
        )
        # initit the superclass with the lambda function
        super(Expected_saturated_distance, self).__init__(f_cost)


def saturated_distance_from_target(
    states_sequence, inputs_sequence, trial_index, target_state, lengthscales, active_dims
):
    """
    The saturated distance defined as:
    1 - exp(-(target_state - states_sequence)^T*(diag(lengthscales^2)^(-1)*(target_state - states_sequence))
    """

    # get state components evaluated in the cost
    active_states = states_sequence[:, :, active_dims]

    # normalize states and targets
    norm_states = active_states / lengthscales
    norm_target = target_state / lengthscales
    # get the square distance
    dist = torch.sum(norm_states**2, dim=2, keepdim=True)
    dist = dist + torch.sum(norm_target**2, dim=1, keepdim=True).transpose(0, 1)
    dist -= 2 * torch.matmul(norm_states, norm_target.transpose(dim0=0, dim1=1))

    cost = 1 - torch.exp(-dist)

    return cost


class Expected_saturated_distance_from_trajectory(Expected_cost):
    """
    Cost function given by the sum of the expected saturated distances from a target trajectory
    """

    def __init__(self, target_traj, lengthscales, flg_var_lengthscales=False, used_indeces=None):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: saturated_distance_from_trajectory(
            x,
            u,
            trial_index,
            target_traj=target_traj,
            lengthscales=lengthscales,
            flg_var_lengthscales=flg_var_lengthscales,
            used_indeces=used_indeces,
        )
        # initit the superclass with the lambda function
        super(Expected_saturated_distance_from_trajectory, self).__init__(f_cost)


def saturated_distance_from_trajectory(
    states_sequence, inputs_sequence, trial_index, target_traj, lengthscales, flg_var_lengthscales, used_indeces
):
    """
    The saturated distance defined as:
    1 - exp(-(target_state - states_sequence)^T*(diag(lengthscales^2)^(-1)*(target_state - states_sequence))
    """
    if used_indeces == None:
        used_indeces = list(range(0, states_sequence.shape[2]))

    # get state components evaluated in the cost
    targets = target_traj.repeat(1, states_sequence.shape[1]).view(states_sequence.shape)
    if flg_var_lengthscales:
        dist = torch.sum(
            ((states_sequence[:, :, used_indeces] - targets[:, :, used_indeces]) / lengthscales[trial_index]) ** 2,
            dim=2,
        )
    else:
        dist = torch.sum(
            ((states_sequence[:, :, used_indeces] - targets[:, :, used_indeces]) / lengthscales) ** 2, dim=2
        )
    cost = 1 - torch.exp(-dist)

    return cost


class Cart_pole_cost(Expected_cost):
    """Cost for the cart pole system:
    target is assumed in the instable equilibrium configuration defined in 'target_state' (target angle [rad], target position [m]).
    """

    def __init__(self, target_state, lengthscales, angle_index, pos_index):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: cart_pole_cost(
            x,
            u,
            trial_index,
            target_state=target_state,
            lengthscales=lengthscales,
            angle_index=angle_index,
            pos_index=pos_index,
        )
        # initit the superclass with the lambda function
        super(Cart_pole_cost, self).__init__(f_cost)


def cart_pole_cost(states_sequence, inputs_sequence, trial_index, target_state, lengthscales, angle_index, pos_index):
    """
    Cost function given by the combination of the saturated distance between |theta| and 'target angle', and between x and 'target position'.
    """
    x = states_sequence[:, :, pos_index]
    theta = states_sequence[:, :, angle_index]

    target_x = target_state[1]
    target_theta = target_state[0]

    return 1 - torch.exp(
        -(((torch.abs(theta) - target_theta) / lengthscales[0]) ** 2) - ((x - target_x) / lengthscales[1]) ** 2
    )


class PenSimCost(Expected_cost):
    """
    Cost function for penicillin fed-batch optimisation.

    Reward signal: maximise penicillin concentration P at every step,
    discounted by gamma^t so that early-batch signal is not swamped by
    the uncertainty that accumulates over 1 150 GP rollout steps.

    States are assumed to be NORMALISED (see envs/pensim_env.py STATE_SCALE).
    P is at state index p_idx (default 6).

    Parameters
    ----------
    p_idx   : int   index of P in the (normalised) state vector
    gamma   : float per-step discount factor (0.999 recommended for 1150 steps)
    """

    def __init__(self, p_idx=6, gamma=0.999):
        f_cost = lambda states, inputs, trial_index: pensim_cost(
            states, inputs, trial_index, p_idx=p_idx, gamma=gamma
        )
        super(PenSimCost, self).__init__(f_cost)


def pensim_cost(states_sequence, inputs_sequence, trial_index, p_idx, gamma):
    """
    Per-step cost = -P_normalised * gamma^t

    states_sequence : [T, N_particles, state_dim]  (normalised)
    Returns         : [T, N_particles]
    """
    T, N, _ = states_sequence.shape
    dtype, device = states_sequence.dtype, states_sequence.device

    # discount vector [T, 1] — earlier steps weighted higher
    t_idx    = torch.arange(T, dtype=dtype, device=device)
    discount = torch.pow(torch.tensor(gamma, dtype=dtype, device=device), t_idx).unsqueeze(1)

    # minimising cost = maximising P
    costs = -states_sequence[:, :, p_idx] * discount   # [T, N]
    return costs


class PenSimCost_v2(Expected_cost):
    """
    Cost for the 13-state, 2-action PenSim formulation.

    Components (all to be minimised):
      1. Terminal P reward  : -w_P * P_norm  at the final step only
      2. PAA band penalty   : quadratic outside [paa_lo_norm, paa_hi_norm] every step
      3. NH3 lower penalty  : quadratic below nh3_lo_norm every step
      4. Action L2          : lambda_u  * ||u_t||^2                every step
      5. Action smoothness  : lambda_du * ||u_t - u_{t-1}||^2      steps 1..T-1

    All indices reference the normalised 13-element state vector in pensim_env.py.

    Parameters
    ----------
    p_idx        : index of P_norm   (default 6)
    paa_idx      : index of PAA_norm (default 10)
    nh3_idx      : index of NH3_norm (default 11)
    paa_lo_norm  : lower PAA bound in normalised units  (default 600/2000  = 0.30)
    paa_hi_norm  : upper PAA bound in normalised units  (default 1800/2000 = 0.90)
    nh3_lo_norm  : lower NH3 bound in normalised units  (default 300/2000  = 0.15)
    w_P          : weight on terminal P reward           (default 1.0)
    lambda_u     : action L2 regularisation weight       (default 1e-3)
    lambda_du    : action smoothness weight              (default 1e-3)
    """

    def __init__(
        self,
        p_idx=6,
        paa_idx=10,
        nh3_idx=11,
        paa_lo_norm=0.30,
        paa_hi_norm=0.90,
        nh3_lo_norm=0.15,
        w_P=1.0,
        lambda_u=1e-3,
        lambda_du=1e-3,
    ):
        f_cost = lambda states, inputs, trial_index: pensim_cost_v2(
            states, inputs, trial_index,
            p_idx=p_idx,
            paa_idx=paa_idx,
            nh3_idx=nh3_idx,
            paa_lo_norm=paa_lo_norm,
            paa_hi_norm=paa_hi_norm,
            nh3_lo_norm=nh3_lo_norm,
            w_P=w_P,
            lambda_u=lambda_u,
            lambda_du=lambda_du,
        )
        super(PenSimCost_v2, self).__init__(f_cost)


def pensim_cost_v2(
    states_sequence, inputs_sequence, trial_index,
    p_idx, paa_idx, nh3_idx,
    paa_lo_norm, paa_hi_norm, nh3_lo_norm,
    w_P, lambda_u, lambda_du,
):
    """
    Per-step cost for the 13-state PenSim formulation.

    states_sequence  : [T, N_particles, state_dim=13]  normalised
    inputs_sequence  : [T, N_particles, input_dim=2]   tanh output in (-1, 1)
    Returns          : [T, N_particles]
    """
    T, N, _ = states_sequence.shape
    dtype, device = states_sequence.dtype, states_sequence.device

    costs = torch.zeros(T, N, dtype=dtype, device=device)

    # 1. Terminal P reward — only at the last timestep
    costs[T - 1] = costs[T - 1] - w_P * states_sequence[T - 1, :, p_idx]

    # 2. PAA band penalty: quadratic outside [lo, hi]
    paa    = states_sequence[:, :, paa_idx]
    paa_lo = torch.tensor(paa_lo_norm, dtype=dtype, device=device)
    paa_hi = torch.tensor(paa_hi_norm, dtype=dtype, device=device)
    costs  = costs + torch.relu(paa_lo - paa) ** 2 + torch.relu(paa - paa_hi) ** 2

    # 3. NH3 lower bound penalty: quadratic below lo
    nh3    = states_sequence[:, :, nh3_idx]
    nh3_lo = torch.tensor(nh3_lo_norm, dtype=dtype, device=device)
    costs  = costs + torch.relu(nh3_lo - nh3) ** 2

    # 4. Action L2 regularisation
    costs = costs + lambda_u * torch.sum(inputs_sequence ** 2, dim=2)

    # 5. Action smoothness (penalise step-to-step changes)
    if T > 1:
        du     = inputs_sequence[1:] - inputs_sequence[:-1]   # [T-1, N, input_dim]
        costs[1:] = costs[1:] + lambda_du * torch.sum(du ** 2, dim=2)

    return costs
