# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def worst_case_quantile_credible_interval(lifespan_sampler, quantile: int | float, num_samples: int = 30000):
    """
    Here we compute the X% quantile credible interval for lifespan. This means that in X% of possible outcomes, the
    lifespan will be longer than the computed bound. Since we have no information about the model, this computation is
    done via sampling.

    Returns
    -------
    float
        The estimated X% quantile lifespan.
    """
    sampled = [lifespan_sampler() for _ in range(num_samples)]
    sampled = np.array(sampled).flatten()

    bound = np.quantile(sampled, 1 - (quantile / 100))
    return bound
