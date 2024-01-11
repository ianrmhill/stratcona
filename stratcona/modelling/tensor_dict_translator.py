# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt

__all__ = ['gen_priors_mapping', 'translate_priors']


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
    return {'prms': map, 'dims': dims}


def translate_priors(priors, map):
    if type(priors) == dict:
        t = np.zeros(map['dims'])
        # Translate from dictionary to tensor
        for i, ltnt in enumerate(map['prms']):
            for j, prm in enumerate(map['prms'][ltnt]):
                t[i][j][:map['prms'][ltnt][prm]] = priors[ltnt][prm]
        return t
    else:
        # Translate from tensor to dictionary
        d = {}
        for i, ltnt in enumerate(map['prms']):
            d[ltnt] = {}
            for j, prm in enumerate(map['prms'][ltnt]):
                d[ltnt][prm] = priors[i][j][:map['prms'][ltnt][prm]]
        return d
