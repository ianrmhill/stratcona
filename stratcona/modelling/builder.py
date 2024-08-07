# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import inspect
from graphlib import TopologicalSorter
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
        if self.dist.rv_op.name in ['normal', 'truncated_normal', 'gamma', 'halfcauchy', 'beta']:
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
        self.pred_args = {}
        self.observes = {}
        self.experiment_params = None
        self.num_experiments = None
        self.samples_per_observation = None
        self.experiment_handle, self.priors_handle, self.observation_handle = None, None, None
        self.experiment_map, self.priors_map, self.observed_map = None, None, None
        self.model_dims = {}
        self.max_var, self.max = None, 0
        self.have_built_model = False

    def add_latent_variable(self, var_name, distribution, prior):
        self.latents[var_name] = LatentVariable(var_name, distribution, prior)

    def add_dependent_variable(self, var_name, compute_func):
        self.dependents[var_name] = compute_func
        self.dep_args[var_name] = inspect.signature(compute_func).parameters.keys()

    def add_lifespan_variable(self, var_name, compute_func):
        self.predictors[var_name] = compute_func
        self.pred_args[var_name] = inspect.signature(compute_func).parameters.keys()

    def gen_lifespan_variable(self, var_name, fail_bounds, field_use_conds = None):
        if field_use_conds is None:
            field_use_conds = {}
        residues = {}
        for dep_var in fail_bounds:
            def residue(time, **kwargs):
                arg_dict = {'time': time}
                for arg in kwargs.keys():
                    if arg in self.dep_args[dep_var]:
                        arg_dict[arg] = kwargs[arg]
                for cond in field_use_conds:
                    if cond != 'time' and cond in self.dep_args[dep_var]:
                        arg_dict[cond] = field_use_conds[cond]
                return abs(fail_bounds[dep_var] - self.dependents[dep_var](**arg_dict))

            residues[dep_var] = residue

        # The overall failure time is based on the first failure to occur out of all the device instances included
        def first_to_fail(**kwargs):
            times = []
            for dep in residues:
                times.append(minimize(residues[dep], kwargs, (np.float64(0.1), np.float64(1e6)), precision=1e-2, log_gold=True))
            return min(times)

        self.predictors[var_name] = first_to_fail
        self.pred_args[var_name] = list(self.latents.keys()) + list(self.dependents.keys())

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

    def define_experiment_params(self, experimental_condition_params: list, simultaneous_experiments: int | list = 1,
                                 samples_per_observation: int | dict = 1):
        self.experiment_params = experimental_condition_params

        # Set up the experimental configuration
        if type(simultaneous_experiments) == int:
            self.num_experiments = simultaneous_experiments
            self.model_dims['experiments'] = [f"exp{i+1}" for i in range(simultaneous_experiments)]
        else:
            self.num_experiments = len(simultaneous_experiments)
            self.model_dims['experiments'] = simultaneous_experiments

        # Set up the number of devices of each observed parameter to measure in each test
        if type(samples_per_observation) == int:
            self.samples_per_observation = {'all': samples_per_observation}
        else:
            self.samples_per_observation = samples_per_observation
            for prm in samples_per_observation:
                self.model_dims[f"num_{prm}"] = np.arange(samples_per_observation[prm])

    def build_model(self, trgt='bed', sample_per_dev=True, ltnt_normalization=None):
        """Builds a model with amortized variability, this model type minimizes the number of latent variable draws"""
        if not self.have_built_model:
            # First check whether any model dimensions still need to be defined
            for var in self.observes:
                if not f"num_{var}" in self.model_dims.keys():
                    if 'all' in self.samples_per_observation.keys():
                        self.model_dims[f"num_{var}"] = np.arange(self.samples_per_observation['all'])
                    else:
                        raise Exception('If number of devices to observe for each observed variable is not defined, then'
                                        'the "all" group must be.')
                if len(self.model_dims[f"num_{var}"]) > self.max:
                    self.max_var = f"num_{var}"
                    self.max = len(self.model_dims[f"num_{var}"])

            # We must configure the latent variable space as ensuring uncertainty is relatively balanced between the
            # set of variables is crucial to avoiding extremely biased optimal experiment design results
            self._normalize_latent_space(ltnt_normalization)

            # Determine dict->tensor mappings then allocate the shared tensor variables used to swap data dynamically
            self.priors_handle = LatentParameterHandler(self.latents)
            self.experiment_handle = ExperimentHandler(self.model_dims['experiments'], self.experiment_params)
            self.observation_handle = ObservationHandler(self.observes, self.samples_per_observation,
                                                     self.model_dims['experiments'])

            self.have_built_model = True

        # TODO: Try to extend shared variable dynamic configuration to the model dimensionality, this would allow for
        #       experiments with different sample sizes of some parameters
        with pymc.Model(coords=self.model_dims) as mdl:
            # First set up the latent independent variables
            ltnts = {}
            ls = {}
            for name, ltnt in self.latents.items():
                if sample_per_dev:
                    ltnts[name] = ltnt.dist(name, dims=self.max_var, **self.priors_handle.get_params(name))
                    ls[name] = pt.tensor.reshape(ltnts[name], (1, self.max))
                else:
                    ltnts[name] = ltnt.dist(name, **self.priors_handle.get_params(name))
                    ls[name] = ltnts[name]

            # Build out the dictionary of experimental conditions
            conds = self.experiment_handle.get_experimental_params()

            # Topological sort the model variables so that some dependent variables can use others as args
            sorted_vars = list(TopologicalSorter(self.dep_args).static_order())
            # Now set up the dependent variables, only providing the necessary inputs to each function
            deps = {}
            for name in sorted_vars:
                if name in self.dependents:
                    args = {arg: val for arg, val in {**ls, **deps, **conds}.items() if arg in self.dep_args[name]}
                    deps[name] = self.dependents[name](**args)

            # Next, set up the observed variables that can be measured during a test
            observes = {}
            for name, obs in self.observes.items():
                means = deps[name]
                # Create the variability parameterization that can be broadcast across experiments and devices observed
                if type(obs['variability']) == str:
                    variability = deps[obs['variability']]
                else:
                    variability = np.full((self.num_experiments, 1), obs['variability'])
                # Extract subtensors for observed variables with fewer observed instances than the maximum
                if name in self.samples_per_observation:
                    means = means[:, :self.samples_per_observation[name]]
                    if type(obs['variability']) == str:
                        variability = variability[:, :self.samples_per_observation[name]]

                if trgt == 'bed':
                    observes[name] = pymc.Normal(name, means, variability,
                                                 shape=(self.num_experiments, self.observation_handle.map[name]),
                                                 dims=('experiments', f"num_{name}"))
                else:
                    observes[f"{name}_obs"] = pymc.Normal(name + '_obs', means, variability,
                                                      observed=self.observation_handle.get_observed(name),
                                                      shape=(self.num_experiments, self.observation_handle.map[name]),
                                                      dims=('experiments', f"num_{name}"))

            # Finally are lifespan variables that are special dependent variables predicting reliability, these are not
            # included for PyMC models because they slow the library algorithms down a lot in certain cases despite not
            # being used for anything
            preds = {}
            if trgt == 'bed':
                for name, pred in self.predictors.items():
                    args = {arg: val for arg, val in {**ls, **deps}.items() if arg in self.pred_args[name]}
                    preds[name] = pymc.Normal(name, pred(**args), 0.2) # FIXME: predictor variability

        return mdl, list(ltnts.values()), list(observes.values()), list(preds.values())
