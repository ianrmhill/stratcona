# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from itertools import cycle
import jax.numpy as jnp
import jax.random as rand


def iter_sampler(possible_vals, cycle_infinitely: bool = True):
    """
    Creates a closure that iterates over a list of values as it is repeatedly called. This allows for deterministic
    lists of values to be used in place of random samplers without appearing any different to the function using them.

    Parameters
    ----------
    possible_vals: iterable
        The full set of possible values that the variable supports.
    cycle_infinitely: bool, optional
        If set to False, the sampler will terminate with a StopIteration exception when all possible values have been
        used. If True (default), the function will wrap and iterate all possible values again, indefinitely.

    Returns
    -------

    """
    iterator = cycle(possible_vals) if cycle_infinitely else iter(possible_vals)

    def _(rng_key):
        return next(iterator)
    return _


def experiment_sampler(design_dimension_vals: list[jnp.ndarray | dict], design_dimension_dists: list = None):
    dist_args = design_dimension_vals
    dim_samplers = design_dimension_dists

    def _(rng_key):
        k1 = rng_key
        items = []
        for i in range(len(dist_args)):
            k1, k2 = rand.split(k1)
            items.append(rand.choice(k2, dist_args[i]) if type(dist_args[i] == jnp.ndarray)\
                else dim_samplers[i](**dist_args[i]))
        return items
    return _


def static_parallel_experiments_sampler(exp_samplers: list):
    """
    Enables sampling of a joint of multiple experiments to allow for static optimization of multiple parallel
    experiments.
    """
    def _():
        return [exp_sampler() for exp_sampler in exp_samplers]
    return _
