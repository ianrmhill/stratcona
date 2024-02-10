# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt
import pymc

from stratcona.assistants.dist_translate import convert_to_categorical
from .variables import define_shared_vars
from .tensor_dict_translator import *

__all__ = ['ModelBuilder']


class LatentVariable():
    def __init__(self, name, distribution, prior_params):
        # Used to scale all latent variables to a common order of magnitude to balance optimization
        self.variance_norm_factor = None
        self.dist = distribution
        self.dist_type = self.get_dist_type()
        if self.dist_type == 'discrete' and self.dist.rv_op.name != 'categorical':
            # Since categorical distributions are by far the easiest to work with for the application, convert every
            # discrete distribution into a categorical via sampling immediately
            prob_weights = convert_to_categorical(self.dist.rv_op.name, prior_params)
            self.dist = pymc.Categorical
            self.user_facing_prms = {'p': prob_weights}
            self.prms = {'p': prob_weights}
        else:
            self.dist = distribution
            self.user_facing_prms = prior_params
            self.prms = prior_params
        self.name = name
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
        if self.dist.rv_op.name in ['normal', 'gamma', 'halfcauchy']:
            return 'continuous'
        elif self.dist.rv_op.name in ['binomial', 'categorical', 'hypergeometric', 'discrete_uniform']:
            return 'discrete'
        else:
            raise NotImplementedError(f"Distribution of type {self.dist.rv_op.name} is not yet supported.")


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
        self.predictors = {}
        self.observes = {}
        self.experiment_params = None
        self.num_experiments = None
        self.samples_per_experiment = None
        self.experiment_handle, self.priors_handle, self.observed_handles = None, None, None
        self.experiment_map, self.priors_map, self.observed_map = None, None, None
        self.dims = {}

    def add_latent_variable(self, var_name, distribution, prior):
        self.latents[var_name] = LatentVariable(var_name, distribution, prior)

    def add_dependent_variable(self, var_name, compute_func):
        # TODO: dependent variables could also be RVs, so instead of a compute function could be a distribution
        # TODO: compute function may not require all latents and experiment params, need to filter like in Gerabaldi
        self.dependents[var_name] = compute_func

    def add_lifespan_variable(self, var_name, compute_func):
        self.predictors[var_name] = compute_func

    def set_variable_observed(self, var_name, variability):
        self.observes[var_name] = {'variability': variability}

    def _normalize_latent_space(self, normalization_strategy = None):
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

    def define_experiment_params(self, prm_list, simultaneous_experiments, samples_per_experiment):
        self.experiment_params = prm_list
        # TODO: Enhance to allow for named experiments
        self.num_experiments = simultaneous_experiments
        self.samples_per_experiment = samples_per_experiment
        self.dims = {'tests': np.arange(simultaneous_experiments), 'devices': np.arange(samples_per_experiment)}

    def _prep_experiments(self):
        # First determine the experiment dict to tensor mapping based on the experiment parameter info
        self.experiment_map = gen_experiment_mapping(self.num_experiments, self.experiment_params)
        return self.experiment_map['dims']

    def _prep_priors(self):
        # First gather the prior values from all the latent variables in the model into a dict
        priors = {}
        for ltnt in self.latents:
            priors[ltnt] = self.latents[ltnt].prms
        # Now determine how to define a tensor that will fit the prior values
        self.priors_map = gen_priors_mapping(priors)
        # Now generate the tensor form for the priors
        return translate_priors(priors, self.priors_map)

    def _prep_observations(self):
        self.observed_map = {}
        for var in self.observes:
            self.observed_map[var] = (self.samples_per_experiment, self.num_experiments)
        return self.observed_map

    def update_priors(self):
        pass

    def build_model(self, ltnt_normalization: str = None):
        # First we must configure the latent variable space as ensuring uncertainty is relatively balanced between the
        # set of variables is crucial to avoiding extremely biased optimal experiment design results
        self._normalize_latent_space(ltnt_normalization)

        # Determine dict->tensor mappings then allocate the shared tensor variables used to swap data dynamically
        pri_t, exp_dims, obs_dims = self._prep_priors(), self._prep_experiments(), self._prep_observations()
        self.experiment_handle, self.priors_handle, self.observed_handles = define_shared_vars(exp_dims, pri_t, obs_dims)

        ### Build the PyMC model now that all elements have been prepped ###
        # TODO: Try to extend shared variable dynamic configuration to the model dimensionality, this would allow for
        #       experiments and parameters with different sample sizes
        with pymc.Model(coords=self.dims) as mdl:
            # First set up the latent independent variables
            priors_mapped = translate_priors(self.priors_handle, self.priors_map)
            ltnts = {}
            for name, ltnt in self.latents.items():
                ltnts[name] = ltnt.dist(name, **priors_mapped[name])

            # Build out the dictionary of experimental conditions
            experiment_params = translate_experiment(self.experiment_handle, self.experiment_map)
            # Add a dimension to allow for broadcasting along the number of devices axis
            for prm in experiment_params:
                experiment_params[prm] = pt.tensor.reshape(experiment_params[prm], (self.num_experiments, 1))

            # Now set up the dependent variables
            deps = {}
            for name, dep in self.dependents.items():
                # FIXME: Currently all dependent variable equations need to accept all latent variables as arguments
                #        or this errors out
                deps[name] = dep(**ltnts, **experiment_params)

            # Next, set up the observed variables that can be measured during a test
            observes = {}
            inf_observes = {}
            for name, obs in self.observes.items():
                # One key challenge with PyMC is that it treats observed and unobserved variables differently in the
                # underlying PyTensor compute graph. We need the unobserved version for the BOED runner, but the
                # observed version for standard inference. We thus define each observed variable twice, once in each
                # form, which allows for both BOED and inference using 'different' variables that are equivalent
                observes[name] = pymc.Normal(name, deps[name], np.full(self.num_experiments, obs['variability']).reshape((self.num_experiments, 1)),
                                             shape=tuple([len(dim) for dim in self.dims.values()]),
                                             dims=tuple(self.dims.keys()))
                inf_observes[name + '_obs'] = pymc.Normal(name + '_obs', deps[name],
                                                          np.full(self.num_experiments, obs['variability']).reshape((self.num_experiments, 1)),
                                                          observed=self.observed_handles[name],
                                                          shape=tuple([len(dim) for dim in self.dims.values()]),
                                                          dims=tuple(self.dims.keys()))

            # Finally are lifespan variables that are special dependent variables predicting reliability
            preds = {}
            for name, pred in self.predictors.items():
                # TODO: make operating conditions a standalone object similar to experiment params or latents rather
                #       than baked into the predictive function
                preds[name] = pred(**ltnts)

        return mdl, list(ltnts.values()), list(observes.values()), list(preds.values())
