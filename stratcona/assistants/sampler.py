# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def experiment_sampler(design_dimension_vals: list[list | dict], design_dimension_dists: list = None):
    dist_args = design_dimension_vals
    dim_samplers = design_dimension_dists

    def _():
        return [np.random.choice(dist_args[i]) if type(dist_args[i]) == list else dim_samplers[i](**dist_args[i]) \
                for i in range(len(dist_args))]
    return _


def static_parallel_experiments_sampler(exp_samplers: list):
    """
    Enables sampling of a joint of multiple experiments to allow for static optimization of multiple parallel
    experiments.
    """
    def _():
        return [exp_sampler() for exp_sampler in exp_samplers]
    return _
