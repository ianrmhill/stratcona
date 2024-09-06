# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import inspect
from graphlib import TopologicalSorter
import numpy as np
import numpyro as npyro
import numpyro.distributions as dists

from stratcona.modelling.relmodel import ReliabilityModel
from stratcona.assistants.dist_translate import convert_to_categorical
from stratcona.engine.minimization import minimize
from .variables import *

__all__ = ['SPMBuilder']


class _HyperLatent():
    def __init__(self, name, distribution, prior_params, transform=None, fixed_prms=None):
        self.dist = distribution
        self.dist_type = self.get_dist_type()
        self.dist_transform = transform
        self.prms = prior_params
        self.fixed_prms = fixed_prms
        self.site_name = name
        self.variance_norm_factor = None

    def compute_prior_entropy(self):
        temp = self.dist(**self.prms)
        return temp.entropy()

    def get_dist_variance(self):
        temp = self.dist(**self.prms)
        return temp.variance

    # FIXME: Any way to generalize?
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
        filter = lambda o: True if inspect.isclass(o) and issubclass(o, dists.Distribution) else False
        continuous_dists = [d[1] for d in inspect.getmembers(dists.continuous, filter)]
        return 'continuous' if self.dist in continuous_dists else 'discrete'


class _Latent():
    def __init__(self, name, dev_dist, dev_prms, chp_var=None, lot_var=None):
        self.site_name = name
        self.dist = dev_dist
        self.dev_prms = dev_prms
        self.lot_var, self.chp_var = lot_var, chp_var


class _LatentNew():
    def __init__(self, name, nom, dev, chp=None, lot=None):
        self.site_name = name
        self.nom = nom
        self.dev = dev
        self.chp = chp
        self.lot = lot


class _Dependent():
    def __init__(self, name, func):
        self.site_name = name
        self.compute = func
        self.requires = list(inspect.signature(self.compute).parameters.keys())

    def build_dep_graph(self, dep_vars):
        nodes = {self.site_name: self.requires}
        for req in self.requires:
            if req in dep_vars:
                nodes |= dep_vars[req].build_dep_graph(dep_vars)
            else:
                nodes[req] = []
        return nodes


class _Measured():
    def __init__(self, name, distribution, dist_prms):
        self.site_name = name
        self.dist = distribution
        self.prms = dist_prms
        self.requires = list(self.prms.values())
        self.dep_graph = {self.site_name: list(self.prms.values())}

    def build_dep_graph(self, dep_vars):
        for req in self.requires:
            if req in dep_vars:
                dep_reqs = dep_vars[req].build_dep_graph(dep_vars)
                self.dep_graph |= dep_reqs
            else:
                self.dep_graph[req] = []
        return self.dep_graph


class _Predictor():
    def __init__(self, name, compute_func, distribution=None, dist_prms=None):
        self.site_name = name
        self.dist = distribution
        self.prms = dist_prms
        self.compute = compute_func
        self.requires = list(inspect.signature(self.compute).parameters.keys())


class SPMBuilder():
    def __init__(self, mdl_name: str):
        self.model_name = mdl_name

        self.hyls = {}
        self.latents = {}
        self.latents_new = {}
        self.dependents = {}
        self.measures = {}
        self.measurement_counts = {}
        self.predictors = {}
        self.predictor_conds = {}
        self.params = {}

    def add_hyperlatent(self, name, distribution, prior, transform=None, fixed_prms=None):
        self.hyls[name] = _HyperLatent(name, distribution, prior, transform, fixed_prms)

    def add_latent(self, name, distribution, dev_prms, chp_var=None, lot_var=None):
        self.latents[name] = _Latent(name, distribution, dev_prms, chp_var, lot_var)

    def add_latent_new(self, name, nom, dev, chp=None, lot=None):
        self.latents_new[name] = _LatentNew(name, nom, dev, chp, lot)

    def add_dependent(self, name, compute_func):
        self.dependents[name] = _Dependent(name, compute_func)

    def add_measured(self, name, distribution, dist_prms, count_per_chp):
        self.measures[name] = _Measured(name, distribution, dist_prms)
        self.measurement_counts[name] = count_per_chp

    def add_params(self, **params):
        for prm in params:
            self.params[prm] = params[prm]

    def add_predictor(self, name, compute_func, pred_conds):
        self.predictors[name] = _Predictor(name, compute_func)
        self.predictor_conds[name] = pred_conds

    def add_predictor_stochastic(self, name, compute_func, distribution, dist_prms, pred_conds):
        self.predictors[name] = _Predictor(name, compute_func, distribution, dist_prms)
        self.predictor_conds[name] = pred_conds

    # TODO: Update
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

    # TODO: Update
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

    def build_model(self, hyl_normalization=None):
        # Perform any hyperlatent normalization
        if hyl_normalization is not None:
            # TODO
            pass

        # Figure out the dependency graph for each measurement
        for obs in self.measures:
            self.measures[obs].build_dep_graph(self.dependents)

        # Construct the probabilistic function that defines the physical model
        def test_model(dims, conds, priors, params, measured=None):
            # First define the hyperlatents that we wish to reduce the entropy of
            hyls = {}
            for hyl in self.hyls:
                if self.hyls[hyl].dist_transform is None:
                    hyls[hyl] = npyro.sample(hyl, self.hyls[hyl].dist(**priors[hyl]))
                else:
                    hyls[hyl] = npyro.sample(hyl, dists.TransformedDistribution(self.hyls[hyl].dist(**priors[hyl]),
                                                                                self.hyls[hyl].dist_transform))

            # Experimental variables are constructed independently for each experiment to ensure each experiment within
            # a test is statistically independent
            samples, ltnts, deps, observes, var_samples = {}, {}, {}, {}, {}
            for exp in dims:
                samples[exp], ltnts[exp], deps[exp], observes[exp], var_samples[exp] = {'lot': {}, 'chp': {}}, {}, {}, {}, {'lot': {}, 'chp': {}, 'dev': {}}

                with npyro.plate(f'{exp}_lot', dims[exp]['lot']):
                    for ltnt in [l for l in self.latents if self.latents[l].lot_var is not None]:
                        samples[exp]['lot'][ltnt] = npyro.sample(f'{ltnt}_{exp}_lot', dists.Normal(0, hyls[self.latents[ltnt].lot_var]))
                    for ltnt in [l for l in self.latents_new if self.latents_new[l].lot is not None]:
                        var_samples[exp]['lot'][ltnt] = npyro.sample(f'{ltnt}_{exp}_lot', dists.TransformedDistribution(
                            dists.Normal(), dists.transforms.AffineTransform(0, hyls[self.latents_new[ltnt].lot])))

                    with npyro.plate(f'{exp}_chp', dims[exp]['chp']):
                        for ltnt in [l for l in self.latents if self.latents[l].chp_var is not None]:
                            samples[exp]['chp'][ltnt] = npyro.sample(f'{ltnt}_{exp}_chp', dists.Normal(0, hyls[self.latents[ltnt].chp_var]))
                        for ltnt in [l for l in self.latents_new if self.latents_new[l].chp is not None]:
                            var_samples[exp]['chp'][ltnt] = npyro.sample(f'{ltnt}_{exp}_chp', dists.TransformedDistribution(
                                dists.Normal(), dists.transforms.AffineTransform(0, hyls[self.latents_new[ltnt].chp])))

                        # Device level behaves differently since each chip may have multiple types of observed circuits,
                        # so this final layer has to have separate configuration per measurement type that is involved
                        for obs in self.measures:
                            # Topological sort the model variables so that some dependent variables can use others as args
                            sorted_vars = list(TopologicalSorter(self.measures[obs].dep_graph).static_order())

                            samples[exp][obs], ltnts[exp][obs], deps[exp][obs], var_samples[exp]['dev'][obs] = {}, {}, {}, {}
                            with npyro.plate(f'{obs}_{exp}_dev', dims[exp][obs]):
                                for ltnt in self.latents:
                                    ltnt_dist_args = {arg: hyls[hyl] for arg, hyl in self.latents[ltnt].dev_prms.items()}
                                    #samples[exp][obs][ltnt] = npyro.sample(f'{ltnt}_{obs}_{exp}_dev', self.latents[ltnt].dist(**ltnt_dist_args))
                                    if not 'loc' in ltnt_dist_args:
                                        ltnt_dist_args['loc'] = 0
                                    samples[exp][obs][ltnt] = npyro.sample(f'{ltnt}_{obs}_{exp}_dev', dists.TransformedDistribution(dists.Normal(0, 1),
                                                                                                                                    dists.transforms.AffineTransform(**ltnt_dist_args)))

                                    # Sum the lot, chp, and device components to get the physical samples for each hierarchical latent
                                    lot_component = samples[exp]['lot'][ltnt] if self.latents[ltnt].lot_var else 0
                                    chp_component = samples[exp]['chp'][ltnt] if self.latents[ltnt].chp_var else 0
                                    ltnts[exp][obs][ltnt] = npyro.deterministic(f'{ltnt}_{obs}_{exp}', lot_component + chp_component + samples[exp][obs][ltnt])

                                for ltnt in self.latents_new:
                                    var_samples[exp]['dev'][obs][ltnt] = npyro.sample(f'{ltnt}_{obs}_{exp}_dev', dists.TransformedDistribution(
                                        dists.Normal(), dists.transforms.AffineTransform(0, hyls[self.latents_new[ltnt].dev])))

                                    nom = hyls[self.latents_new[ltnt].nom]
                                    dev = var_samples[exp]['dev'][obs][ltnt]
                                    chp = var_samples[exp]['chp'][ltnt] if self.latents_new[ltnt].chp else 0
                                    lot = var_samples[exp]['lot'][ltnt] if self.latents_new[ltnt].lot else 0
                                    ltnts[exp][obs][ltnt] = npyro.deterministic(f'{ltnt}_{obs}_{exp}', nom + dev + chp + lot)

                                # Now compute the dependent variables, only providing the necessary inputs to each function
                                for dep in [n for n in sorted_vars if n in self.dependents]:
                                    args = {arg: val for arg, val in {**ltnts[exp][obs], **deps[exp][obs], **conds[exp], **params}.items() if arg in self.dependents[dep].requires}
                                    deps[exp][obs][dep] = npyro.deterministic(f'{dep}_{obs}_{exp}', self.dependents[dep].compute(**args))

                                # Next, create the observed variables measured in the experiment
                                # Observed variable distributions can be parameterized in terms of dependent variables or params
                                concated = {**ltnts[exp][obs], **deps[exp][obs], **conds[exp], **params}
                                obs_dist_args = {arg: concated[val] for arg, val in self.measures[obs].prms.items()}
                                measd = measured[exp][obs] if measured is not None else None
                                observes[exp][obs] = npyro.sample(f'{obs}_{exp}', self.measures[obs].dist(**obs_dist_args), obs=measd)

        # Construct the probabilistic function that defines the physical model
        def lifespan_model(dims, conds, priors, params, measured=None):
            # First define the hyperlatents that represent the beliefs/uncertainty in the model
            hyls = {}
            for hyl in self.hyls:
                if self.hyls[hyl].dist_transform is None:
                    hyls[hyl] = npyro.sample(hyl, self.hyls[hyl].dist(**priors[hyl]))
                else:
                    hyls[hyl] = npyro.sample(hyl, dists.TransformedDistribution(self.hyls[hyl].dist(**priors[hyl]),
                                                                                self.hyls[hyl].dist_transform))
            # The lifespan sample can consist of some number of lots and chips
            samples, ltnts, deps, observes, var_samples = {'lot': {}, 'chp': {}}, {}, {}, {}, {'lot': {}, 'chp': {}, 'dev': {}}
            with npyro.plate(f'pred_lot', dims['lot']):
                for ltnt in [l for l in self.latents if self.latents[l].lot_var is not None]:
                    samples['lot'][ltnt] = npyro.sample(f'{ltnt}_pred_lot', dists.Normal(0, hyls[self.latents[ltnt].lot_var]))
                for ltnt in [l for l in self.latents_new if self.latents_new[l].lot is not None]:
                    var_samples['lot'][ltnt] = npyro.sample(f'{ltnt}_pred_lot', dists.TransformedDistribution(
                        dists.Normal(), dists.transforms.AffineTransform(0, hyls[self.latents_new[ltnt].lot])))

                with npyro.plate(f'pred_chp', dims['chp']):
                    for ltnt in [l for l in self.latents if self.latents[l].chp_var is not None]:
                        samples['chp'][ltnt] = npyro.sample(f'{ltnt}_pred_chp', dists.Normal(0, hyls[self.latents[ltnt].chp_var]))
                    for ltnt in [l for l in self.latents_new if self.latents_new[l].chp is not None]:
                        var_samples['chp'][ltnt] = npyro.sample(f'{ltnt}_pred_chp', dists.TransformedDistribution(
                            dists.Normal(), dists.transforms.AffineTransform(0, hyls[self.latents_new[ltnt].chp])))

                    # Device level behaves differently since each chip may have multiple types of observed circuits,
                    # so this final layer has to have separate configuration per measurement type that is involved
                    for obs in self.measures:
                        # Topological sort the model variables so that some dependent variables can use others as args
                        sorted_vars = list(TopologicalSorter(self.measures[obs].dep_graph).static_order())

                        samples[obs], ltnts[obs], deps[obs], var_samples['dev'][obs] = {}, {}, {}, {}
                        with npyro.plate(f'{obs}_pred_dev', dims[obs]):
                            for ltnt in self.latents:
                                ltnt_dist_args = {arg: hyls[hyl] for arg, hyl in self.latents[ltnt].dev_prms.items()}
                                samples[obs][ltnt] = npyro.sample(f'{ltnt}_{obs}_pred_dev', self.latents[ltnt].dist(**ltnt_dist_args))

                                # Sum the lot, chp, and device components to get the physical samples for each hierarchical latent
                                lot_component = samples['lot'][ltnt] if self.latents[ltnt].lot_var else 0
                                chp_component = samples['chp'][ltnt] if self.latents[ltnt].chp_var else 0
                                ltnts[obs][ltnt] = npyro.deterministic(f'{ltnt}_{obs}_pred', lot_component + chp_component + samples[obs][ltnt])

                            for ltnt in self.latents_new:
                                var_samples['dev'][obs][ltnt] = npyro.sample(f'{ltnt}_{obs}_pred_dev', dists.TransformedDistribution(
                                    dists.Normal(), dists.transforms.AffineTransform(0, hyls[self.latents_new[ltnt].dev])))

                                nom = hyls[self.latents_new[ltnt].nom]
                                dev = var_samples['dev'][obs][ltnt]
                                chp = var_samples['chp'][ltnt] if self.latents_new[ltnt].chp else 0
                                lot = var_samples['lot'][ltnt] if self.latents_new[ltnt].lot else 0
                                ltnts[obs][ltnt] = npyro.deterministic(f'{ltnt}_{obs}_pred', nom + dev + chp + lot)

                            # Now compute the dependent variables, only providing the necessary inputs to each function
                            for dep in [n for n in sorted_vars if n in self.dependents]:
                                args = {arg: val for arg, val in {**ltnts[obs], **deps[obs], **conds, **params}.items() if arg in self.dependents[dep].requires}
                                deps[obs][dep] = npyro.deterministic(f'{dep}_{obs}_pred', self.dependents[dep].compute(**args))

                            # Next, create the observed variables measured in the experiment
                            # Observed variable distributions can be parameterized in terms of dependent variables or params
                            concated = {**ltnts[obs], **deps[obs], **conds, **params}
                            obs_dist_args = {arg: concated[val] for arg, val in self.measures[obs].prms.items()}
                            observes[obs] = npyro.sample(f'{obs}_pred', self.measures[obs].dist(**obs_dist_args))

                    # Finally are lifespan variables that are special dependent variables predicting reliability
                    preds, pred_funcs = {}, {}
                    for pred in self.predictors:
                        args = {arg: val for arg, val in {**observes, **conds, **params}.items() if arg in self.predictors[pred].requires}
                        if self.predictors[pred].dist is None:
                            pred_funcs[pred] = npyro.deterministic(f'{pred}', self.predictors[pred].compute(**args))
                        else:
                            pred_funcs[pred] = npyro.deterministic(f'{pred}_lf', self.predictors[pred].compute(**args))
                            concated = {**pred_funcs, **params}
                            obs_dist_args = {arg: concated[val] for arg, val in self.predictors[pred].prms.items()}
                            actual = measured[pred] if measured is not None else None
                            preds[pred] = npyro.sample(f'{pred}', self.predictors[pred].dist(**obs_dist_args), obs=actual)

        # Assemble the contextual information for the model needed to work with the defined SPMs
        hyl_priors = {hyl: self.hyls[hyl].prms for hyl in self.hyls}
        hyl_info = {hyl: {'dist': self.hyls[hyl].dist, 'fixed': self.hyls[hyl].fixed_prms} for hyl in self.hyls}
        ltnt_subsample_site_names = [f'{ltnt}_dev' for ltnt in self.latents]
        ltnt_subsample_site_names.extend([f'{ltnt}_chp' for ltnt in self.latents if self.latents[ltnt].chp_var is not None])
        ltnt_subsample_site_names.extend([f'{ltnt}_lot' for ltnt in self.latents if self.latents[ltnt].lot_var is not None])

        return ReliabilityModel(self.model_name, test_model, lifespan_model, self.params,
                                self.hyls.keys(), hyl_priors, hyl_info,
                                self.latents.keys(), ltnt_subsample_site_names,
                                self.measures.keys(), self.measurement_counts,
                                self.predictors.keys(), self.predictor_conds)
