# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp


def qx_lbci(samples: jnp.ndarray, x: int | float):
    """
    Here we compute the X% quantile credible interval for lifespan. This means that in X% of possible outcomes, the
    lifespan will be longer than the computed bound. Since we have no information about the model, this computation is
    done via sampling.

    Returns
    -------
    float
        The estimated X% quantile lifespan.
    """
    return jnp.quantile(samples, 1 - (x / 100))


def qx_lbci_l(samples: jnp.ndarray, x: int | float):
    """
    Here we compute the X% quantile credible interval for lifespan. This means that in X% of possible outcomes, the
    lifespan will be longer than the computed bound. Since we have no information about the model, this computation is
    done via sampling.
    """
    sampled = samples.copy().flatten()
    sampled = sampled[sampled > 0]
    l_lbci = jnp.quantile(jnp.log(sampled), 1 - (x / 100))
    return jnp.exp(l_lbci)


def qx_hdcr(samples: jnp.ndarray, x: int | float, num_bins: int = 1000):
    """
    The highest density interval can be useful for situations in which we aren't worried about liability of edge cases
    with poor reliability and prefer to find a general estimate of the likely lifespan for a product.

    Parameters
    ----------
    lifespan_sampler
    interval_size
    num_samples

    Returns
    -------

    """
    # To find the highest density credible region for an arbitrary distribution that may be non-parametric, we will use
    # an estimate based on numerical sampling
    sampled = samples.copy().flatten()
    num_samples = sampled.size

    # We bin the samples into intervals, then keep adding the intervals with the highest counts until we pass the target
    start, stop = jnp.min(sampled), jnp.max(sampled)
    bins = jnp.linspace(start, stop, num_bins)
    bin_len = bins[1] - bins[0]
    densities, intervals = jnp.histogram(sampled, bins, density=True)
    # Sort the bins from highest density to lowest (and their corresponding intervals)
    i_order = jnp.flip(jnp.argsort(densities))
    sorted_density = densities[i_order]
    sorted_intervals = intervals[i_order]
    # Add the largest bins until the interval size is surpassed
    i = 0
    while jnp.sum(sorted_density[:i] * bin_len) < (x / 100):
        i += 1
    # Now turn all the little interval chunks into a set of tuples specifying the highest density region
    intervals = sorted_intervals[:i]
    intervals = jnp.sort(intervals)
    # Assemble all the individual bins into contiguous intervals where possible
    start = None
    region = []
    for i in range(len(intervals)):
        if start is None:
            start = intervals[i]
            # Special handling for case where x\% of samples are within a single bin
            if len(intervals) == 1:
                region.append((start, start + bin_len))
        else:
            # Multiply bin_len by 1.1 during comparison to avoid rounding of floats from impacting the comparison
            if i >= len(intervals) - 1 or (intervals[i+1] - intervals[i]) > (1.1 * bin_len):
                region.append((start, intervals[i] + bin_len))
                start = None

    # Before returning, perform a quick analysis to determine if the settings were reasonable
    # 1. Check that the credible region isn't divided into many little segments due to poor sampling or binning
    if len(region) > 4:
        print(f"Highest density credible region has {len(region)} modes, it is likely that this is an error and that"
              "either the bin count should be reduced or the number of samples should be increased.")
    # 2. Check that the actual region doesn't cover significantly (1%) more or less density than desired
    sample_count = 0
    for i in range(len(region)):
        sample_count += ((region[i][0] < sampled) & (sampled < region[i][1])).sum()
    if jnp.abs((sample_count / num_samples) - (x / 100)) > 0.01:
        print(f"Highest density credible region estimate covers {sample_count / num_samples}% of samples, it is likely"
              "that the bin count should be increased to reduce this estimation error.")
    return region


def qx_hdcr_l(samples: jnp.ndarray, x: int | float, num_bins: int = 1000):
    """
    The highest density interval can be useful for situations in which we aren't worried about liability of edge cases
    with poor reliability and prefer to find a general estimate of the likely lifespan for a product.
    """
    # To find the highest density credible region for an arbitrary distribution that may be non-parametric, we will use
    # an estimate based on numerical sampling
    sampled = samples.copy().flatten()
    # Cannot have negative values in a log bin setting
    sampled = jnp.log(sampled[sampled > 0])
    num_samples = sampled.size

    # We bin the samples into intervals, then keep adding the intervals with the highest counts until we pass the target
    start, stop = jnp.min(sampled), jnp.max(sampled)
    bins = jnp.linspace(start, stop, num_bins)
    bin_len = bins[1] - bins[0]
    densities, intervals = jnp.histogram(sampled, bins, density=True)
    # Sort the bins from highest density to lowest (and their corresponding intervals)
    i_order = jnp.flip(jnp.argsort(densities))
    sorted_density = densities[i_order]
    sorted_intervals = intervals[i_order]
    # Add the largest bins until the interval size is surpassed
    i = 0
    while jnp.sum(sorted_density[:i] * bin_len) < (x / 100):
        i += 1
    # Now turn all the little interval chunks into a set of tuples specifying the highest density region
    intervals = sorted_intervals[:i]
    intervals = jnp.sort(intervals)
    # Assemble all the individual bins into contiguous intervals where possible
    start = None
    region = []
    for i in range(len(intervals)):
        if start is None:
            start = intervals[i]
            # Special handling for case where x\% of samples are within a single bin
            if len(intervals) == 1:
                region.append((jnp.exp(start), jnp.exp(start + bin_len)))
        else:
            # Multiply bin_len by 1.1 during comparison to avoid rounding of floats from impacting the comparison
            if i >= len(intervals) - 1 or (intervals[i+1] - intervals[i]) > (1.1 * bin_len):
                region.append((jnp.exp(start), jnp.exp(intervals[i] + bin_len)))
                start = None

    # Before returning, perform a quick analysis to determine if the settings were reasonable
    # 1. Check that the credible region isn't divided into many little segments due to poor sampling or binning
    if len(region) > 4:
        print(f"Highest density credible region has {len(region)} modes, it is likely that this is an error and that"
              "either the bin count should be reduced or the number of samples should be increased.")
    # 2. Check that the actual region doesn't cover significantly (1%) more or less density than desired
    sample_count = 0
    for i in range(len(region)):
        sample_count += ((region[i][0] < jnp.exp(sampled)) & (jnp.exp(sampled) < region[i][1])).sum()
    if jnp.abs((sample_count / num_samples) - (x / 100)) > 0.01:
        print(f"Highest density credible region estimate covers {sample_count / num_samples}% of samples, it is likely"
              "that the bin count should be increased to reduce this estimation error.")
    return region


def mttf(samples: jnp.ndarray):
    return samples.sum() / samples.size
