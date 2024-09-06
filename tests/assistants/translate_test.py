# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax.random as rand
import jax.numpy as jnp
import scipy.stats as sci_stats
import numpyro.distributions as dists

import matplotlib.pyplot as plt

from stratcona.assistants.dist_translate import npyro_to_scipy


def test_npyro_scipy_equivalence():
    loc, scale = 0.04, 0.3
    lognorm_samples = sci_stats.lognorm.rvs(scale, scale=loc, size=1000)
    key = rand.key(234)
    compare_samples = dists.LogNormal(jnp.log(loc), scale).sample(key, (1000,))

    fig, axes = plt.subplots(1, 2)
    axes[0].hist(lognorm_samples, bins=50)
    axes[1].hist(compare_samples, bins=50)
    plt.show()
    assert jnp.mean(lognorm_samples) == jnp.mean(compare_samples)


def test_transforms():
    key = rand.key(597)
    ns = (10_000,)
    loc, scale = 2, 1
    base_samples = dists.Normal(loc, scale).sample(key, ns)
    transform_samples = dists.TransformedDistribution(dists.Normal(loc, scale),
                                                      [dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.01)]).sample(key, ns)
    double_tf_samples = dists.TransformedDistribution(dists.Normal(loc, scale),
                                                      [dists.transforms.AffineTransform(0, 0.01), dists.transforms.SoftplusTransform()]).sample(key, ns)
    fig, axes = plt.subplots(1, 3)
    axes[0].hist(base_samples, bins=50)
    axes[1].hist(transform_samples, bins=50)
    axes[2].hist(double_tf_samples, bins=50)
    plt.show()


def test_softplus_scaled():
    key = rand.key(597)
    ns = (10_000,)
    loc, scale = 2, 1
    base_samples = dists.Normal(loc, scale).sample(key, ns)
    transform_samples = dists.TransformedDistribution(dists.Normal(loc, scale),
                                                      dists.transforms.SoftplusTransform()).sample(key, ns)
    double_tf_samples = dists.TransformedDistribution(dists.Normal(loc, scale),
                                                      [dists.transforms.AffineTransform(0, 1), dists.transforms.SoftplusTransform()]).sample(key, ns)
    fig, axes = plt.subplots(1, 3)
    axes[0].hist(base_samples, bins=50)
    axes[1].hist(transform_samples, bins=50)
    axes[2].hist(double_tf_samples, bins=50)
    plt.show()


def test_hyper_dist_viability():
    key = rand.key(597)
    ns = (10_000,)
    loc, scale = 4.2, 0
    s1 = dists.StudentT(0.1, loc, scale).sample(key, ns)
    s2 = dists.SoftLaplace(loc, scale).sample(key, ns)
    s3 = dists.Normal(loc, scale).sample(key, ns)
    fig, axes = plt.subplots(1, 3)
    axes[0].hist(s1, bins=50)
    axes[1].hist(s2, bins=50)
    axes[2].hist(s3, bins=50)
    plt.show()
