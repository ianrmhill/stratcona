# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import inspect
import numpy as np
import pytensor as pt
import pymc

from stratcona.assistants.dist_translate import convert_to_categorical
from stratcona.engine.minimization import minimize
from .variables import *

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
            case 'beta':
                a, b = self.prms['alpha'], self.prms['beta']
                return (a * b) / (((a + b)**2) * (a + b + 1))
            case _:
                raise NotImplementedError

    def variance_to_prms(self, new_variance):
        match self.dist.rv_op.name:
            case 'normal':
                self.prms['sigma'] = new_variance**0.5
            case 'beta':
                a, b = self.prms['alpha'], self.prms['beta']
                mean = a / (a + b)
                v = ((mean * (1 - mean)) / new_variance) - 1
                self.prms['alpha'] = mean * v
                self.prms['beta'] = (1 - mean) * v
            case _:
                raise NotImplementedError

    def get_dist_type(self):
        if self.dist.rv_op.name in ['normal', 'gamma', 'halfcauchy', 'beta']:
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
        self.dep_args = {}
        self.predictors = {}
        self.observes = {}
        self.experiment_params = None
        self.num_experiments = None
        self.samples_per_experiment = None
        self.experiment_handle, self.priors_handle, self.observed_handles = None, None, None
        self.experiment_map, self.priors_map, self.observed_map = None, None, None
        self.model_dims = {}

    def add_latent_variable(self, var_name, distribution, prior):
        self.latents[var_name] = LatentVariable(var_name, distribution, prior)

    def add_dependent_variable(self, var_name, compute_func):
        # TODO: dependent variables could also be RVs, so instead of a compute function could be a distribution
        # TODO: compute function may not require all latents and experiment params, need to filter like in Gerabaldi
        self.dependents[var_name] = compute_func
        self.dep_args[var_name] = inspect.signature(compute_func).parameters.keys()

    def add_lifespan_variable(self, var_name, compute_func):
        self.predictors[var_name] = compute_func

    def gen_lifespan_variable(self, var_name, fail_bounds, field_use_conds):
        residues = {}
        for dep_var in fail_bounds:
            def residue(time, **ltnts):
                arg_dict = {'time': time}
                # TODO: args dict may also need other dependent variables
                for ltnt in ltnts.keys():
                    if ltnt in self.dep_args[dep_var]:
                        arg_dict[ltnt] = ltnts[ltnt]
                for cond in field_use_conds:
                    if cond != 'time' and cond in self.dep_args[dep_var]:
                        arg_dict[cond] = field_use_conds[cond]
                return abs(fail_bounds[dep_var] - self.dependents[dep_var](**arg_dict))

            residues[dep_var] = residue

        # The overall failure time is based on the first failure to occur out of all the device instances included
        def first_to_fail(**ltnts):
            times = []
            for dep in residues:
                # TODO: Evaluate whether a log-scale time minimization strategy is feasible
                times.append(minimize(residues[dep], ltnts, (np.float64(0.0), np.float64(1e5)), precision=1e-2))
            return min(times)

        self.predictors[var_name] = first_to_fail

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
        if normalization_strategy is None:
            return
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

    def define_experiment_params(self, experimental_condition_params: list, simultaneous_experiments: int | list,
                                 samples_per_experiment: int | dict):
        self.experiment_params = experimental_condition_params

        # Set up the experimental configuration
        if type(simultaneous_experiments) == int:
            self.num_experiments = simultaneous_experiments
            self.model_dims['experiments'] = [f"exp{i+1}" for i in range(simultaneous_experiments)]
        else:
            self.num_experiments = len(simultaneous_experiments)
            self.model_dims['experiments'] = simultaneous_experiments

        # Set up the number of devices of each observed parameter to measure in each test
        if type(samples_per_experiment) == int:
            self.samples_per_experiment = {'all': samples_per_experiment}
        else:
            self.samples_per_experiment = samples_per_experiment
            for prm in samples_per_experiment:
                self.model_dims[f"num_{prm}"] = np.arange(samples_per_experiment[prm])

    def build_model(self, ltnt_normalization: str = None):
        # First check whether any model dimensions still need to be defined
        for var in self.observes:
            if not f"num_{var}" in self.model_dims.keys():
                if 'all' in self.samples_per_experiment.keys():
                    self.model_dims[f"num_{var}"] = np.arange(self.samples_per_experiment['all'])
                else:
                    raise Exception('If number of devices to observe for each observed variable is not defined, then'
                                    'the "all" group must be.')

        # We must configure the latent variable space as ensuring uncertainty is relatively balanced between the
        # set of variables is crucial to avoiding extremely biased optimal experiment design results
        self._normalize_latent_space(ltnt_normalization)

        # Determine dict->tensor mappings then allocate the shared tensor variables used to swap data dynamically
        self.priors_handle = LatentParameterHandler(self.latents)
        self.experiment_handle = ExperimentHandler(self.model_dims['experiments'], self.experiment_params)
        self.observation_handle = ObservationHandler(self.observes, self.samples_per_experiment, self.model_dims['experiments'])

        ### Build the PyMC model now that all elements have been prepped ###
        # TODO: Try to extend shared variable dynamic configuration to the model dimensionality, this would allow for
        #       experiments with different sample sizes of some parameters
        with pymc.Model(coords=self.model_dims) as mdl:
            # First set up the latent independent variables
            ltnts = {}
            ls = {}
            for name, ltnt in self.latents.items():
                ltnts[name] = ltnt.dist(name, **self.priors_handle.get_params(name))
                # Reshape the variables for broadcasting, can't do using the 'shape' argument for dist as it causes
                # certain distributions to error out when compiling
                ls[name] = pt.tensor.reshape(ltnts[name], (1, 1))

            # Build out the dictionary of experimental conditions
            experiment_params = self.experiment_handle.get_experimental_params()
            # Now set up the dependent variables, only providing the necessary inputs to each function
            deps = {}
            for name, dep in self.dependents.items():
                arg_dict = {arg: val for arg, val in {**ls, **experiment_params}.items() if arg in self.dep_args[name]}
                deps[name] = dep(**arg_dict)

            # Next, set up the observed variables that can be measured during a test
            observes = {}
            inf_observes = {}
            for name, obs in self.observes.items():
                # Create the variability parameterization that can be broadcast across experiments and devices observed
                variability = np.full((self.num_experiments, 1), obs['variability'])

                # One key challenge with PyMC is that it treats observed and unobserved variables differently in the
                # underlying PyTensor compute graph. We need the unobserved version for the BOED runner, but the
                # observed version for standard inference. We thus define each observed variable twice, once in each
                # form, which allows for both BOED and inference using 'different' variables that are equivalent
                observes[name] = pymc.Normal(name, deps[name], variability,
                                             shape=(self.num_experiments, self.observation_handle.map[name]),
                                             dims=('experiments', f"num_{name}"))
                inf_observes[f"{name}_obs"] = pymc.Normal(name + '_obs', deps[name], variability,
                                                          observed=self.observation_handle.get_observed(name),
                                                          shape=(self.num_experiments, self.observation_handle.map[name]),
                                                          dims=('experiments', f"num_{name}"))

            # Finally are lifespan variables that are special dependent variables predicting reliability
            preds = {}
            for name, pred in self.predictors.items():
                # TODO: make operating conditions a standalone object similar to experiment params or latents rather
                #       than baked into the predictive function
                preds[name] = pymc.Normal(name, pred(**ls), 0.2)

        return mdl, list(ltnts.values()), list(observes.values()), list(preds.values())
