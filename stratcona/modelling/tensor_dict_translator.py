# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt

__all__ = ['gen_priors_mapping', 'translate_priors', 'gen_experiment_mapping',
           'translate_experiment', 'gen_observation_mapping', 'translate_observations']


def gen_priors_mapping(priors):
    # The map contains the names of each latent variable and parameters that specify the variable distribution with
    # values that detail how many values are needed for each parameter
    map = {}
    # Need to know the overall dimensions of the tensor form for the priors, many of the elements may end up unused
    # Dimensions are: num_latents x max_parameters_per_latent x max_values_per_parameter
    dims = [1, 1, 1]
    dims[0] = len(priors.keys())
    for ltnt in priors:
        if len(priors[ltnt].keys()) > dims[1]:
            dims[1] = len(priors[ltnt].keys())
        map[ltnt] = {}
        for prm in priors[ltnt]:
            if type(priors[ltnt][prm]) in [int, float]:
                map[ltnt][prm] = 1
            else:
                # Some distribution parameters are lists (such as the pymc.Categorical 'p' parameter)
                if len(priors[ltnt][prm]) > dims[2]:
                    dims[2] = len(priors[ltnt][prm])
                map[ltnt][prm] = len(priors[ltnt][prm])
    return {'prms': map, 'dims': tuple(dims)}


def translate_priors(priors, map):
    if type(priors) == dict:
        t = np.zeros(map['dims'])
        # Translate from dictionary to tensor
        for i, ltnt in enumerate(map['prms']):
            for j, prm in enumerate(map['prms'][ltnt]):
                try:
                    t[i][j][:map['prms'][ltnt][prm]] = priors[ltnt][prm]
                except ValueError:
                    # This is used for categorical RVs, as the number of possible values take may have decreased
                    # in the posterior, thus we need to pad out zeros to keep the probability weights array the same
                    # length as before
                    pad_len = map['prms'][ltnt][prm] - len(priors[ltnt][prm])
                    t[i][j][:map['prms'][ltnt][prm]] = np.pad(priors[ltnt][prm], (0, pad_len))
        return t
    else:
        # Translate from tensor to dictionary
        d = {}
        for i, ltnt in enumerate(map['prms']):
            d[ltnt] = {}
            for j, prm in enumerate(map['prms'][ltnt]):
                d[ltnt][prm] = priors[i][j][:map['prms'][ltnt][prm]]
        return d


def gen_experiment_mapping(experiments: int | list, experiment_prms):
    # Tensor dimensions are <number of experiments> x <number of stress/experiment parameters>
    if type(experiments) == int:
        dims = (experiments, len(experiment_prms))
        return {'prms': experiment_prms, 'dims': dims}
    else:
        # Passing experiments as a list instead of a count allows for named experiments
        dims = (len(experiments), len(experiment_prms))
        return {'exps': experiments, 'prms': experiment_prms, 'dims': dims}


def translate_experiment(experiments, map, target='internal'):
    # There are three representations of the experiment parameters needed: 1. tensor form for computation,
    # 2. internal dict of params to array of experiment values for passing to functions, 3. human readable dict
    if type(experiments) == dict:
        t = np.zeros(map['dims'])
        if 'exps' in map.keys():
            for i, exp in enumerate(map['exps']):
                for j, prm in enumerate(map['prms']):
                    t[i, j] = experiments[exp][prm]
        else:
            for i in range(map['dims'][0]):
                for j, prm in enumerate(map['prms']):
                    t[i, j] = experiments[f"exp{i+1}"][prm]
        return t
    elif target == 'internal':
        e = {}
        for i, prm in enumerate(map['prms']):
            e[prm] = experiments[:, i]
        return e
    else:
        e = {}
        if 'exps' in map.keys():
            for i, exp in enumerate(map['exps']):
                e[exp] = {}
                for j, prm in enumerate(map['prms']):
                    e[exp][prm] = experiments[i, j]
        else:
            for i in range(map['dims'][0]):
                e[f"exp{i+1}"] = {}
                for j, prm in enumerate(map['prms']):
                    e[f"exp{i+1}"][prm] = experiments[i, j]
        return e


def gen_observation_mapping(observation_counts, experiments: int | list):
    map = {'prms': observation_counts}
    # Tensor dimensions are:
    # <number of observed variables> x <max number of instances of variable observed> x <number of experiments>
    if type(experiments) == int:
        dims = [len(observation_counts.keys()), 1, experiments]
    else:
        # Optional support for named experiments
        dims = [len(observation_counts.keys()), 1, len(experiments)]
        map['exps'] = experiments
    for var in observation_counts:
        if observation_counts[var] > dims[1]:
            dims[1] = observation_counts[var]
    map['dims'] = tuple(dims)
    return map


def translate_observations(observations, map, target='internal'):
    if type(observations) == dict:
        t = np.zeros(map['dims'])
        # Translate from dictionary to tensor
        for i, var in enumerate(map['prms']):
            if type(observations[var]) == dict:
                for j, exp in enumerate(map['exps']):
                    t[i, :map['prms'][var], j] = observations[var][exp]
            else:
                t[i, :map['prms'][var], :] = observations[var]
        return t
    else:
        # Translate from tensor to dictionary
        d = {}
        for i, var in enumerate(map['prms']):
            if target != 'internal' and 'exps' in map.keys():
                d[var] = {}
                for j, exp in enumerate(map['exps']):
                    d[var][exp] = observations[i, :map['prms'][var], j]
            else:
                d[var] = observations[i, :map['prms'][var], :]
        return d
