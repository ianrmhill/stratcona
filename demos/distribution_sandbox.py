# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import jax.random as rand
import numpyro as npyro
import numpyro.distributions as dists

import seaborn as sb
from matplotlib import pyplot as plt

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stratcona


def check_dist():
    dist = dists.Delta(v=0.7)

    key = rand.key(2394723)
    samples = dist.sample(key, (10_000,))

    ### Plot the distribution ###
    sb.set_context('notebook')
    sb.set_theme(style='ticks', font='Times New Roman')
    fig, p = plt.subplots(1, 1)

    p.hist(samples, 100, density=True, histtype='stepfilled',
           label=f'Mean: {round(dist.mean, 4)}, Var: {round(dist.variance, 4)}')
    p.legend()

    plt.show()


def metrics_bimodal_comparison():
    dist1 = dists.Weibull(scale=70.5, concentration=8.4)
    dist2 = dists.LogNormal(loc=3.5, scale=0.21)
    k1, k2 = rand.split(rand.key(2023590827))
    samples = jnp.concatenate((dist1.sample(k1, (500_000,)), dist2.sample(k2, (300_000,))))

    ### Compute metrics ###
    region = stratcona.engine.metrics.qx_hdcr(samples, 90, num_bins=400)
    r_as_text = ''
    for i, segment in enumerate(region):
        r_as_text += f'[{round(float(segment[0]), 2)}, {round(float(segment[1]), 2)}]'
        if i < len(region) - 1:
            r_as_text += ', '
    lifespan = stratcona.engine.metrics.qx_lbci(samples, 99.9)
    mttf = stratcona.engine.metrics.mttf(samples)

    ### Plot the distribution ###
    sb.set_context('notebook')
    sb.set_theme(style='ticks', font='Times New Roman')
    fig, p = plt.subplots(1, 1)

    # Plot shaded regions first so they are behind the histogram visually
    for i, segment in enumerate(region):
        p.axvspan(segment[0], segment[1], 0, 1, color='lightblue', alpha=0.3)

    p.hist(samples, 300, density=True, color='grey', histtype='stepfilled')

    p.axvline(mttf, 0, 1, color='green', linestyle='dashed', label=f'MTTF: {round(float(mttf), 2)}')
    p.axvline(lifespan, 0, 1, color='orange', linestyle='dashed', label=f'Q99.9%-LBCI: {round(float(lifespan), 2)}')
    # Now plot the Q90%-HDCR
    for i, segment in enumerate(region):
        lbl = f'Q90%-HDCR:\n{r_as_text}' if i == 0 else None
        p.axvline(segment[0], 0, 1, color='lightblue', linestyle='dotted', label=lbl)
        p.axvline(segment[1], 0, 1, color='lightblue', linestyle='dotted')

    p.legend(loc='upper right')
    p.set_xlabel('Failure Time (years)', fontsize='medium')
    p.set_ylabel('Probability Density')

    plt.show()


if __name__ == '__main__':
    metrics_bimodal_comparison()
