# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt
import pymc

import stratcona


def test_discrete():
    mb = stratcona.ModelBuilder(mdl_name='Discrete Degradation')

    def param_reduction(a, b, c, temp, time):
        return a * np.exp((b * temp) / 1000) * (time ** (c / 10))

    mb.add_latent_variable('a', pymc.HyperGeometric, {'N': 30, 'k': 5, 'n': 8})
    mb.add_latent_variable('b', pymc.Categorical, {'p': [0.2, 0.3, 0.3, 0.2]})
    mb.add_latent_variable('c', pymc.Categorical, {'p': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]})

    mb.define_experiment_params(['temp', 'time'], simultaneous_experiments=1, samples_per_experiment=100)

    mb.add_dependent_variable('deg', param_reduction)
    mb.set_variable_observed('deg', variability=0.3)

    # TODO: mb.add_lifespan_variable('fail_point', ?, ?, ?)

    tm = stratcona.TestDesignManager(mb)

    tm.estimate_reliability()
    tm.determine_best_test()
    tm.infer_model()
    tm.estimate_reliability()
