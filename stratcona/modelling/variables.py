# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt

__all__ = ['define_shared_vars']


def define_shared_vars(exp_dims: tuple | np.ndarray, prior_dims: tuple | np.ndarray, obs_dims: tuple | np.ndarray):
    """
    By placing the experimental conditions, prior distribution info, and observed data within shared tensor
    variables we are able to change these values without having to then redefine the model and recompile the compute
    functions needed for BOED and inference algorithms.

    Parameters
    ----------
    exp_dims: tuple or ndarray
        Either the experimental values themselves, or the dimensions. Dim #1: number of experiments, dim #2: number of
        conditions (e.g. temperature, voltage, humidity) that each experiment defines.
    prior_dims: tuple or ndarray
        Either the prior values or the dimensions of the values. Dim #1: number of latent variables, dim #2: max number
        of parameters used to define the latent variable distributions (e.g., a Normal distribution needs 2 parameters,
        while a Triangular needs 3 to define the distribution).
    obs_dims: tuple or ndarray
        Either the observed data or the dimensions of the observations. Dim #1: number of observed variables, dim #2:
        number of experiments, dim #3: number of devices sampled per experiment.
    """
    exp_handle = pt.shared(np.zeros(exp_dims) if type(exp_dims) == tuple else exp_dims)
    prior_handle = pt.shared(np.zeros(prior_dims) if type(prior_dims) == tuple else prior_dims)
    # TODO: What if we measure different numbers of devices for different observed variables?
    obs_handle = pt.shared(np.zeros(obs_dims) if type(obs_dims) == tuple else obs_dims)
    return exp_handle, prior_handle, obs_handle
