# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt
import pymc

__all__ = ['define_shared_vars', 'extract_experiment_params']


def define_shared_vars(exp_dims: tuple | np.ndarray, prior_dims: tuple | np.ndarray):
    exp_handle = pt.shared(np.zeros(exp_dims) if type(exp_dims) == tuple else exp_dims)
    prior_handle = pt.shared(np.zeros(prior_dims) if type(prior_dims) == tuple else prior_dims)
    return exp_handle, prior_handle


def extract_experiment_params(exp_handle, prm_names):
    # Retrieve the experiment values from the shared variable which will be in a numpy array format
    exps = exp_handle.get_value()
    num_prms = len(exps[0])
    # We place the parameter values, bundled across any experiment multiples, into a dict for easy keyword args passing
    extracted = {}
    for prm in range(num_prms):
        extracted[prm_names[prm]] = exps[:, prm]
    return extracted
