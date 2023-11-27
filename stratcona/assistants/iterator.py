# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from itertools import cycle

__all__ = ['iter_sampler']


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

    def _():
        return next(iterator)
    return _
