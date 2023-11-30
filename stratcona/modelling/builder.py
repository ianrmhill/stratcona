# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt
import pymc

from .variables import define_shared_vars

__all__ = ['ModelBuilder']


class LatentVariable():
    def __init__(self, name, distribution, prior_params):
        # Used to scale all latent variables to a common order of magnitude to balance optimization
        self.variance_norm_factor = None
        self.dist = distribution
        self.dist_type = self.get_dist_type()
        self.name = name
        self.user_facing_prms = prior_params
        self.prms = prior_params
        self.entropy = None

    def compute_prior_entropy(self):
        pass

    def get_dist_variance(self):
        match self.dist.rv_op.name:
            case 'normal':
                return self.prms['sigma']**2
            case _:
                raise NotImplementedError

    def variance_to_prms(self, new_variance):
        match self.dist.rv_op.name:
            case 'normal':
                self.prms['sigma'] = new_variance**0.5
            case _:
                raise NotImplementedError

    def get_dist_type(self):
        if self.dist.rv_op.name in ['normal', 'gamma']:
            return 'continuous'
        elif self.dist.rv_op.name in ['binomial']:
            return 'discrete'
        else:
            actual_type = type(self.dist)
            raise NotImplementedError


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
        self.discrete_latent_entropy = []
        self.dependents = {}
        self.observes = {}
        self.experiment_params = None
        self.num_experiments = None
        self.samples_per_experiment = None
        self.experiment_handle, self.priors_handle = None, None
        self.dims = {}

    def add_latent_variable(self, var_name, distribution, prior):
        self.latents[var_name] = LatentVariable(var_name, distribution, prior)

    def add_dependent_variable(self, var_name, compute_func):
        # TODO: dependent variables could also be RVs, so instead of a compute function could be a distribution
        # TODO: compute function may not require all latents and experiment params, need to filter like in Gerabaldi
        self.dependents[var_name] = compute_func

    def set_variable_observed(self, var_name, variability):
        self.observes[var_name] = {'variability': variability}

    def _normalize_latent_space(self):
        """
        Normalizes the variance of the latent variable space to make the entropy approximately equal for each variable.
        This reduces the bias of EIG calculation towards variables with higher entropy. Note that this is not the best
        way to do this, as the relative importance of each variable to the output quantity(ies) of interest depends on
        the sensitivity of those quantities with respect to the latent variables. This method will be updated in the
        future to use a better approach based on sensitivity analysis once output quantities of interest and sensitivity
        analysis have been implemented (TODO).
        """
        # 1. Check if any latents are discrete, as their entropy cannot be adjusted, unlike the differential entropy
        if len(self.discrete_latent_entropy) > 0:
            target_variance = 1.0 # TODO: Get mean entropy of discrete latent variables, target is the variance of a Gaussian with that differential entropy
        else:
            target_variance = 1.0 # TODO: Get average variance of continuous latents

        # 2. Adjust all latent variables by scaling factors to match the average and update helpful properties so the
        # user can view the variable as though the transformation was not made (for ease of use)
        for ltnt in self.latents.values():
            if ltnt.dist_type == 'continuous':
                ltnt.variance_to_prms(target_variance)

    def extract_priors(self):
        prior_vals = self.priors_handle.get_value()
        extracted = {}
        for i, var in enumerate(self.latents):
            extracted[var] = {}
            for j, prm in enumerate(self.latents[var].prms):
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
        # First we must configure the latent variable space as ensuring uncertainty is relatively balanced between the
        # set of variables is crucial to avoiding extremely biased optimal experiment design results
        self._normalize_latent_space()
        priors_formatted = np.array([list(var.prms.values()) for var in self.latents.values()])

        exp_dims = (self.num_experiments, len(self.experiment_params))
        self.experiment_handle, self.priors_handle = define_shared_vars(exp_dims, priors_formatted)

        # Build the PyMC model now that all elements have been prepped
        with pymc.Model(coords=self.dims) as mdl:
            priors_mapped = self.extract_priors()
            ltnts = {}
            for name, ltnt in self.latents.items():
                ltnts[name] = ltnt.dist(name, **priors_mapped[name])

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
