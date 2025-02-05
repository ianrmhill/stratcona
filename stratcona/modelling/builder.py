# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import inspect
from graphlib import TopologicalSorter
from typing import Self

import jax.numpy as jnp
import numpyro as npyro
import numpyro.distributions as dists
from numpyro.distributions import TransformedDistribution as TrDist
from numpyro.distributions.transforms import AffineTransform as AffineTr

from stratcona.modelling.relmodel import ReliabilityModel
from stratcona.engine.minimization import minimize


def unity(x): return x


class _HyperLatent():
    def __init__(self, name, distribution, prior_params, transform=None, fixed_prms=None, scaling=1.0):
        self.dist = distribution
        self.is_continuous = self.is_continuous()
        # Determine the scaling factor applied to the variable, if any
        self.tf = transform if transform is not None else unity
        self.tf_inv = transform._inverse if transform is not None else unity
        self.prms = prior_params
        self.fixed_prms = fixed_prms
        self.site_name = name
        self.variance_norm_factor = None
        self.scale_factor = scaling

    def compute_prior_entropy(self):
        temp = self.dist(**self.prms)
        return temp.entropy()

    def get_dist_variance(self):
        temp = self.dist(**self.prms)
        return temp.variance

    def is_continuous(self):
        filtered = lambda o: True if inspect.isclass(o) and issubclass(o, dists.Distribution) else False
        continuous_dists = [d[1] for d in inspect.getmembers(dists.continuous, filtered)]
        return True if self.dist in continuous_dists else False


class _Latent():
    def __init__(self, name, nom, dev=None, chp=None, lot=None):
        self.site_name = name
        self.nom = nom
        self.dev = dev
        self.chp = chp
        self.lot = lot


class _Composite():
    def __init__(self, name: str, requires: list[str]):
        self.site_name = name
        self.requires = requires
        self.dep_graph = None
        self.deps_sorted = None

    def build_dep_graph(self, dep_vars: dict[str: Self]):
        # Constructs a graph as a dictionary of nodes with edges in the format expected by the TopologicalSorter class
        nodes = {self.site_name: self.requires}
        for req in self.requires:
            if req in dep_vars:
                # Recurse down the tree of composite variables that will be defined
                nodes |= dep_vars[req].build_dep_graph(dep_vars)
            else:
                # Is a latent variable that is guaranteed to be defined before any dependent variables
                nodes[req] = []
        return nodes


class _CDeterministic(_Composite):
    def __init__(self, name, func):
        self.compute = func
        super().__init__(name, list(inspect.signature(self.compute).parameters.keys()))


class _CStochastic(_Composite):
    def __init__(self, name, dist, prms):
        self.dist = dist
        self.prms = prms
        super().__init__(name, list(self.prms.values()))


class SPMBuilder():
    def __init__(self, mdl_name: str):
        self.model_name = mdl_name

        self.params = {}
        self.hyls = {}
        self.latents = {}
        self.intermediates = {}
        self.observes = {}
        self.predictors = {}
        self.observable_counts = {}
        self.fail_criteria = {}

    def add_params(self, **params):
        for prm in params:
            self.params[prm] = params[prm]

    def add_hyperlatent(self, name, distribution, prior, transform=None, fixed_prms=None):

        self.hyls[name] = _HyperLatent(name, distribution, prior, transform, fixed_prms)

    def add_latent(self, name, nom, dev, chp=None, lot=None):
        self.latents[name] = _Latent(name, nom, dev, chp, lot)

    def add_intermediate(self, name, compute_func):
        self.intermediates[name] = _CDeterministic(name, compute_func)

    def add_intermediate_stochastic(self, name, distribution, dist_prms):
        self.intermediates[name] = _CStochastic(name, distribution, dist_prms)

    def add_observed(self, name, distribution, dist_prms, count_per_chp):
        self.observes[name] = _CStochastic(name, distribution, dist_prms)
        self.observable_counts[name] = count_per_chp

    def add_predictor(self, name, compute_func):
        self.predictors[name] = _CDeterministic(name, compute_func)
        self.observable_counts[name] = 1

    def add_predictor_stochastic(self, name, distribution, dist_prms):
        self.predictors[name] = _CStochastic(name, distribution, dist_prms)
        self.observable_counts[name] = 1

    def add_fail_criterion(self, name, compute_func):
        self.fail_criteria[name] = _CDeterministic(name, compute_func)

    # TODO: Update
    def gen_fail_criterion(self, var_name, fail_bounds, field_use_conds = None):
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

    def build_model(self):
        # Figure out the dependency graph for each observable
        for obs in self.observes | self.predictors:
            obs_info = self.observes[obs] if obs in self.observes else self.predictors[obs]
            obs_info.dep_graph = obs_info.build_dep_graph(self.intermediates)
            # Sort the model variables so that some intermediate variables can use others as args
            obs_info.deps_sorted = list(TopologicalSorter(obs_info.dep_graph).static_order())

        # Construct the probabilistic function that defines the physical model
        def spm(dims, conds, priors, params, measured=None, compute_predictors=True):
            hyls_and_prms, samples, ltnts, meds, observes, fail_criteria = params.copy(), {}, {}, {}, {}, {}
            # First define the hyper-latents that we wish to reduce the entropy of
            for name, hyl in self.hyls.items():
                if hyl.tf is unity:
                    hyls_and_prms[name] = npyro.sample(name, hyl.dist(**priors[name]))
                else:
                    hyls_and_prms[name] = npyro.sample(name, TrDist(hyl.dist(**priors[name]), hyl.tf))

            # Experimental variables are constructed independently for each experiment to ensure that each experiment
            # within a test is statistically independent
            for exp in dims:
                samples[exp], ltnts[exp], meds[exp], observes[exp], fail_criteria[exp] = {'lot': {}, 'chp': {}, 'dev': {}}, {}, {}, {}, {}

                with npyro.plate(f'{exp}_lot', dims[exp]['lot']):
                    for ltnt in [l for l in self.latents if self.latents[l].lot is not None]:
                        samples[exp]['lot'][ltnt] = npyro.sample(f'{exp}_{ltnt}_lot_ls', TrDist(
                            dists.Normal(), AffineTr(0, hyls_and_prms[self.latents[ltnt].lot])))

                    with npyro.plate(f'{exp}_chp', dims[exp]['chp']):
                        for ltnt in [l for l in self.latents if self.latents[l].chp is not None]:
                            samples[exp]['chp'][ltnt] = npyro.sample(f'{exp}_{ltnt}_chp_ls', TrDist(
                                dists.Normal(), AffineTr(0, hyls_and_prms[self.latents[ltnt].chp])))

                        # Device level behaves differently since each chip may have multiple types of observed circuits,
                        # so this final layer has to have separate configuration per measurement type that is involved
                        obs_vars = self.observes | self.predictors if compute_predictors else self.observes
                        for obs in obs_vars:
                            samples[exp]['dev'][obs], ltnts[exp][obs], meds[exp][obs] = {}, {}, {}
                            obs_info = self.observes[obs] if obs in self.observes else self.predictors[obs]
                            # Allow for experiment definitions to override the number of devices observed. In general
                            # this is not used but allows for comparisons of experiments with varied observable counts
                            obs_dev_count = dims[exp][obs] if obs in dims[exp] else self.observable_counts[obs]

                            with npyro.plate(f'{exp}_{obs}_dev', obs_dev_count):
                                for ltnt in [l for l in self.latents if self.latents[l].dev is not None]:
                                    samples[exp]['dev'][obs][ltnt] = npyro.sample(f'{exp}_{obs}_{ltnt}_dev_ls', TrDist(
                                        dists.Normal(), AffineTr(0, hyls_and_prms[self.latents[ltnt].dev])))

                                for ltnt in self.latents:
                                    # Merge all the multilevel variations to form the final sample values for the latent
                                    nom = hyls_and_prms[self.latents[ltnt].nom]
                                    dev = samples[exp]['dev'][obs][ltnt] if self.latents[ltnt].dev else 0
                                    chp = samples[exp]['chp'][ltnt] if self.latents[ltnt].chp else 0
                                    lot = samples[exp]['lot'][ltnt] if self.latents[ltnt].lot else 0
                                    ltnts[exp][obs][ltnt] = npyro.deterministic(f'{exp}_{obs}_{ltnt}',
                                                                                nom + dev + chp + lot)

                                # Compute intermediate variables; only provide the necessary inputs to each function
                                for med in [n for n in obs_info.deps_sorted if n in self.intermediates]:
                                    args = {arg: val for arg, val in {**ltnts[exp][obs], **meds[exp][obs], **conds[exp], **hyls_and_prms}.items() if arg in self.intermediates[med].requires}
                                    try:
                                        if type(self.intermediates[med]) is _CStochastic:
                                            dist_args = {arg: args[val] for arg, val in obs_info.prms.items()}
                                            meds[exp][obs][med] = npyro.sample(f'{exp}_{obs}_{med}', self.intermediates[med].dist(**dist_args))
                                        else:
                                            meds[exp][obs][med] = npyro.deterministic(f'{exp}_{obs}_{med}', self.intermediates[med].compute(**args))
                                    except KeyError as e:
                                        if e.args[0] == 'fn':
                                            raise Exception(f'Intermediate variable {med} compute function does not return!')
                                        raise e

                                # Next, define the observed or predictor variable measured in the experiment
                                args = {arg: val for arg, val in {**ltnts[exp][obs], **meds[exp][obs], **conds[exp], **hyls_and_prms}.items() if arg in obs_info.requires}
                                measd = measured[exp][obs] if measured is not None else None
                                if type(obs_info) is _CStochastic:
                                    dist_args = {arg: args[val] for arg, val in obs_info.prms.items()}
                                    observes[exp][obs] = npyro.sample(f'{exp}_{obs}', obs_info.dist(**dist_args), obs=measd)
                                else:
                                    observes[exp][obs] = npyro.deterministic(f'{exp}_{obs}', obs_info.compute(**args))

                        # Special predictor variables that reduce observed values down to a single yes/no failure status
                        # per chip, cannot take intermediate variables as inputs since those are sampled independently
                        # per observed variable
                        if compute_predictors:
                            for criterion in self.fail_criteria:
                                args = {arg: val for arg, val in {**observes[exp], **conds[exp], **hyls_and_prms}.items() if arg in self.fail_criteria[criterion].requires}
                                fail_criteria[exp][criterion] = npyro.deterministic(f'{exp}_{criterion}', self.fail_criteria[criterion].compute(**args))

        # Assemble the contextual information for the model needed to work with the defined SPMs
        hyl_priors = {hyl: self.hyls[hyl].prms for hyl in self.hyls}
        hyl_info = {hyl: {'dist': self.hyls[hyl].dist, 'fixed': self.hyls[hyl].fixed_prms,
                          'transform': self.hyls[hyl].tf, 'transform_inv': self.hyls[hyl].tf_inv} for hyl in self.hyls}
        ltnt_subsample_site_names = [f'{ltnt}_dev_ls' for ltnt in self.latents if self.latents[ltnt].dev is not None]
        ltnt_subsample_site_names.extend([f'{ltnt}_chp_ls' for ltnt in self.latents if self.latents[ltnt].chp is not None])
        ltnt_subsample_site_names.extend([f'{ltnt}_lot_ls' for ltnt in self.latents if self.latents[ltnt].lot is not None])

        return ReliabilityModel(self.model_name, spm, self.params,
                                list(self.hyls.keys()), hyl_priors, hyl_info,
                                list(self.latents.keys()), ltnt_subsample_site_names,
                                list(self.observes.keys()), self.observable_counts,
                                list(self.predictors.keys()), list(self.fail_criteria.keys()))
