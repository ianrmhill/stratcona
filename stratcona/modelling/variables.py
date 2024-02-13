# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt

__all__ = ['LatentParameterHandler', 'ExperimentHandler', 'ObservationHandler']


# TODO: The internal optimizations of pytensor can be dangerous. Because of the internal optimizations, if you set
#       shared tensor values in specific ways, entire sections of the compute graph can be optimized away incorrectly
#       since internally PyTensor thinks these are constants. Essentially you can't compile the compute function
#       without risking errors until you've set non-unitary values for the shared variable values. Need to ensure this
#       behaviour for the user so they don't have to worry about it.
class ExperimentHandler():
    """
    Class to manage the PyTensor shared variables needed to dynamically change experimental stress conditions without
    recompiling the model. Dimensions are <number of experiments> x <number of condition parameters>
    """
    def __init__(self, experiments, condition_params):
        self.exp_order = tuple(experiments)
        self.cond_order = tuple(condition_params)
        self.dims = (len(experiments), len(condition_params))
        # Create the shared variable tensor, initialize all conditional parameters to 2 to avoid compilation functions
        # from optimizing away operations if the parameters happen to be unitary operations
        self.tensor = pt.shared(np.full(self.dims, 2.0))

    def set_experimental_params(self, conditions):
        self.tensor.set_value(self._d_to_t(conditions))

    def get_experimental_params(self):
        return self._t_to_d()

    def _t_to_d(self):
        d = {}
        # We get the conditions from all experiments as that is what is required by the PyMC model
        for i, cond in enumerate(self.cond_order):
            # Also add a dimension to the end to enable broadcasting across the device sample dimension
            d[cond] = pt.tensor.reshape(self.tensor[:, i], (self.dims[0], 1))
        return d

    def _d_to_t(self, conditions):
        t = np.zeros(self.dims)
        for i, exp in enumerate(self.exp_order):
            for j, cond in enumerate(self.cond_order):
                t[i, j] = conditions[exp][cond]
        return t


class LatentParameterHandler():
    """
    Class to manage the PyTensor shared variables needed to dynamically change model priors without recompiling the
    model. Each latent variable gets a shared variable that stores the distribution parameterization with a
    dimensionality of <num_params> x <max_vals_per_param>. For example, a Normal distribution with mu and sigma would be
    2x1, a categorical distribution with N possible values is 1xN, and a distribution with prm1: [x, y, z] and prm2: [w]
    would be 2x3. Note that this dimensionality must be fixed at declaration, so a categorical distribution cannot be
    dynamically updated with more possible values, only shrink with unused values getting probability 0.
    """
    def __init__(self, latent_vars):
        self.tensors = {}
        self.map = {}
        self.dims = {}
        for ltnt in latent_vars:
            self.map[ltnt] = {}
            num_prms = len(latent_vars[ltnt].prms.keys())
            max_prm_len = 1
            for prm, val in latent_vars[ltnt].prms.items():
                if type(val) in [int, float]:
                    self.map[ltnt][prm] = 1
                # Some distribution parameters are lists (such as the pymc.Categorical 'p' parameter)
                else:
                    if len(val) > max_prm_len:
                        max_prm_len = len(val)
                    self.map[ltnt][prm] = len(val)
            self.dims[ltnt] = (num_prms, max_prm_len)
            # Declare the PyTensor shared variable with the prior values set accordingly
            self.tensors[ltnt] = pt.shared(self._d_to_t(ltnt, latent_vars[ltnt].prms))

    def get_params(self, ltnt):
        return self._t_to_d(ltnt)

    def set_params(self, vals):
        for ltnt in self.map:
            self.tensors[ltnt].set_value(self._d_to_t(ltnt, vals[ltnt]))

    def _t_to_d(self, ltnt):
        d = {}
        for i, prm in enumerate(self.map[ltnt]):
            d[prm] = self.tensors[ltnt][i, :self.map[ltnt][prm]]
        return d

    def _d_to_t(self, ltnt, vals):
        t = np.zeros(self.dims[ltnt])
        for i, prm in enumerate(self.map[ltnt]):
            try:
                t[i, :self.map[ltnt][prm]] = vals[prm]
            except ValueError:
                # This is used for categorical RVs, as the number of possible values take may have decreased, thus we
                # need to pad out zeros to keep the probability weights array the same length as before
                pad_len = self.map[ltnt][prm] - len(vals[prm])
                t[i, :self.map[ltnt][prm]] = np.pad(vals[prm], (0, pad_len))
        return t


class ObservationHandler():
    def __init__(self, observed_vars, devices_per_var, experiments):
        self.tensors = {}
        self.map = {}
        self.exp_order = tuple(experiments)
        self.dims = {}
        for var in observed_vars:
            self.map[var] = devices_per_var[var] if var in devices_per_var.keys() else devices_per_var['all']
            self.dims[var] = (len(self.exp_order), self.map[var])
            self.tensors[var] = pt.shared(np.full(self.dims[var], 2.0))

    def get_observed(self, var):
        return self._t_to_d(var)

    def set_observed(self, observations):
        self._d_to_t(observations)

    def _t_to_d(self, var):
        return self.tensors[var]

    def _d_to_t(self, observations):
        ts = {}
        for var in self.map.keys():
            ts[var] = np.zeros(self.dims[var])
            for i, exp in enumerate(self.exp_order):
                ts[var][i, :] = observations[exp][var]
        return ts
