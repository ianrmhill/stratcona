# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def pymc_to_scipy(dist_type):
    match dist_type:
        case 'normal':
            return 'norm'
        case _:
            return dist_type


def pymc_to_numpy(dist_type):
    match dist_type:
        case _:
            return dist_type


def convert_to_categorical(dist, prms, num_samples: int = 30_000):
    # Count the number of occurrences of each value then normalize to get the posterior probabilities
    np_dist = pymc_to_numpy(dist)
    sampled = getattr(np.random, np_dist)(**prms, size=num_samples)
    return np.bincount(sampled) / num_samples
