# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
from datetime import timedelta

from gerabaldi.models.reports import TestSimReport
import gracefall


def worst_case_quantile_credible_region(lifespan_sampler, interval_size: int | float, num_samples: int = 30000,
                                        plot_sampled=True, sampler_style='vectorized'):
    """
    Here we compute the X% quantile credible interval for lifespan. This means that in X% of possible outcomes, the
    lifespan will be longer than the computed bound. Since we have no information about the model, this computation is
    done via sampling.

    Returns
    -------
    float
        The estimated X% quantile lifespan.
    """
    if sampler_style == 'vectorized':
        sampled = lifespan_sampler(num_samples)
    else:
        sampled = [lifespan_sampler() for _ in range(num_samples)]
    sampled = jnp.array(sampled).flatten()

    # Optionally generate a plot of all the sampled failure times
    if plot_sampled:
        report = TestSimReport(name='Sampled Failure Times')
        exp_samples = sampled.reshape((1, 1, -1))
        as_dataframe = TestSimReport.format_measurements(exp_samples, 'exp0', timedelta(), 0)
        report.add_measurements(as_dataframe)
        gracefall.static.gen_violinplot(report.measurements)

    # Determine the failure time at the specified quantile (representing the effective product lifespan)
    bound = jnp.quantile(sampled, 1 - (interval_size / 100))
    return bound


def highest_density_credible_region(lifespan_sampler, interval_size: int | float,
                                    num_samples: int = 30000, num_bins: int = 1000):
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
    sampled = [lifespan_sampler() for _ in range(num_samples)]
    sampled = jnp.array(sampled).flatten()

    # We bin the samples into intervals, then keep adding the intervals with the highest counts until we pass the target
    bins = jnp.linspace(jnp.min(sampled), jnp.max(sampled), num_bins)
    bin_len = bins[1] - bins[0]
    densities, intervals = jnp.histogram(sampled, bins, density=True)
    # Sort the bins from highest density to lowest (and their corresponding intervals)
    i_order = jnp.flip(jnp.argsort(densities))
    sorted_density = densities[i_order]
    sorted_intervals = intervals[i_order]
    # Add the largest bins until the interval size is surpassed
    i = 0
    while jnp.sum(sorted_density[:i] * bin_len) < (interval_size / 100):
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
        else:
            if i >= len(intervals) - 1 or not round(intervals[i+1], 5) == round(intervals[i] + bin_len, 5):
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
    if jnp.abs((sample_count / num_samples) - (interval_size / 100)) > 0.01:
        print(f"Highest density credible region estimate covers {sample_count / num_samples}% of samples, it is likely"
              "that the bin count should be increased to reduce this estimation error.")
    return region


def qx_lbci(samples, x: int | float):
    """
    Here we compute the X% quantile credible interval for lifespan. This means that in X% of possible outcomes, the
    lifespan will be longer than the computed bound. Since we have no information about the model, this computation is
    done via sampling.

    Returns
    -------
    float
        The estimated X% quantile lifespan.
    """
    sampled = samples.copy()
    sampled = jnp.array(sampled).flatten()
    num_samples = sampled.size

    # Determine the failure time at the specified quantile (representing the effective product lifespan)
    bound = jnp.quantile(sampled, 1 - (x / 100))
    return bound


def qx_hdcr(samples, x: int | float, num_bins: int = 1000):
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
    sampled = samples.copy()
    sampled = jnp.array(sampled).flatten()
    num_samples = sampled.size

    # We bin the samples into intervals, then keep adding the intervals with the highest counts until we pass the target
    bins = jnp.linspace(jnp.min(sampled), jnp.max(sampled), num_bins)
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
        else:
            # Multiply bin_len by 1.1 to avoid rounding of floats from impacting the comparison
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


def mttf(samples):
    sampled = samples.copy()
    sampled = jnp.array(sampled).flatten()
    num_samples = sampled.size

    mean = sampled.sum() / num_samples
    return mean
