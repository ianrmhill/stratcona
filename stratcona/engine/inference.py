# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import arviz
import scipy


def fit_latent_params_to_posterior_samples(latents: dict, idata: arviz.InferenceData):
    """
    After PyMC runs a sampling algorithm we end up with many samples from the posterior. What PyMC does not do, however,
    is update the parameterized latent variables based on the posterior. Here we do this manually.

    Returns
    -------

    """
    post_data = idata.posterior
    posterior_params = {}
    for ltnt in latents:
        sampled = post_data[ltnt].values.flatten()
        # TODO: Need to get the correct mapping from PyMC to Scipy and then back, will need a mapping function for this
        dist_to_fit = latents[ltnt].dist.rv_op.name
        scipy_dist = getattr(scipy.stats, dist_to_fit)
        fitted = scipy.stats.fit(scipy_dist, sampled)
        print(fitted)
        posterior_params[ltnt] = {'mu': fitted.params[0], 'sigma': fitted.params[1]}

    return posterior_params
