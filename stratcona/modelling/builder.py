# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt
import pymc

from .variables import define_shared_vars

__all__ = ['ModelBuilder']


class LatentVariable():
    def __init__(self):
        # Note that for continuous variables, this should be Jaynes' limiting density of discrete points definition to
        # ground the absolute entropy, not the continuous version of Shannon's definition
        self.entropy = None
        # Used to scale all latent variables to a common order of magnitude to balance optimization
        self.norm_factor = None


class ModelBuilder():
    """
    This is used as a wrapper/constructor around the PyMC Model class to allow for the following features:
        1. Dynamic prior and observed values, experiments, and model dimensionality to avoid variant model redefinition
        2. Automatic normalization and tracking of latent variable uncertainty for optimization process transparency
        3. Analysis of discrete vs. continuous model features used to select appropriate BOED and inference algorithms
    These features are all used to make optimal experiment design and sequential inference more streamlined.
    """
    def __init__(self, mdl_name: str):
        self.model_name = mdl_name
        self.latents = {}
        self.dependents = {}
        self.observes = {}
        self.experiment_params = None
        self.num_experiments = None
        self.samples_per_experiment = None
        self.experiment_handle, self.priors_handle = None, None
        self.dims = {}

    def add_latent_variable(self, var_name, distribution, prior):
        # TODO: IMMEDIATE: No normalization or anything yet, implement normalization and Jaynes' entropy computation
        self.latents[var_name] = {'dist': distribution, 'prior': prior}

    def add_dependent_variable(self, var_name, compute_func):
        # TODO: dependent variables could also be RVs, so instead of a compute function could be a distribution
        # TODO: compute function may not require all latents and experiment params, need to filter like in Gerabaldi
        self.dependents[var_name] = compute_func

    def set_variable_observed(self, var_name, variability):
        self.observes[var_name] = {'variability': variability}

    def extract_priors(self):
        prior_vals = self.priors_handle.get_value()
        extracted = {}
        for i, var in enumerate(self.latents):
            extracted[var] = {}
            for j, prm in enumerate(self.latents[var]['prior']):
                extracted[var][prm] = prior_vals[i][j]
        return extracted

    def update_priors(self):
        pass

    def define_experiment_params(self, prm_list, simultaneous_experiments, samples_per_experiment):
        self.experiment_params = prm_list
        self.num_experiments = simultaneous_experiments
        self.samples_per_experiment = samples_per_experiment
        self.dims = {'devices': np.arange(samples_per_experiment), 'tests': np.arange(simultaneous_experiments)}

    def extract_experiment_params(self):
        # Retrieve the experiment values from the shared variable which will be in a numpy array format
        exps = self.experiment_handle.get_value()
        num_prms = len(self.experiment_params)
        # We place the parameter values, bundled across multiple experiments, into a dict for easy keyword args passing
        extracted = {}
        for i in range(num_prms):
            extracted[self.experiment_params[i]] = exps[:, i]
        return extracted

    def build_model(self):
        priors_formatted = np.array([list(var['prior'].values()) for var in self.latents.values()])
        exp_dims = (self.num_experiments, len(self.experiment_params))
        self.experiment_handle, self.priors_handle = define_shared_vars(exp_dims, priors_formatted)

        # Build the PyMC model now that all elements have been prepped
        with pymc.Model(coords=self.dims) as mdl:
            priors_mapped = self.extract_priors()
            ltnts = {}
            for name, ltnt in self.latents.items():
                ltnts[name] = ltnt['dist'](name, **priors_mapped[name])

            experiment_params = self.extract_experiment_params()

            deps = {}
            for name, dep in self.dependents.items():
                deps[name] = dep(**ltnts, **experiment_params)

            observes = {}
            for name, obs in self.observes.items():
                observes[name] = pymc.Normal(name, deps[name], np.full(self.num_experiments, obs['variability']),
                                             shape=tuple([len(dim) for dim in self.dims.values()]),
                                             dims=tuple(self.dims.keys()))

        return mdl, list(ltnts.values()), list(observes.values())
