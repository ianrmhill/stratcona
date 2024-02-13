# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt
import pymc

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TEMP
from matplotlib import pyplot as plt

import stratcona


def discrete_wearout_demo():
    mb = stratcona.ModelBuilder(mdl_name='Discrete Degradation')

    def happy_little_linear_degradation(a, b, c, d, vdd, temp, time):
        return -1 * ((a / 10) + 0.8) * (b * vdd) * (c * (temp / 100)) * (d * (time / 1000))

    mb.add_latent_variable('a', pymc.Categorical, {'p': [0.1, 0.1, 0.3, 0.3, 0.1, 0.1]})
    mb.add_latent_variable('b', pymc.Categorical, {'p': [0.0, 0.2, 0.3, 0.3, 0.2, 0.0]})
    mb.add_latent_variable('c', pymc.Binomial, {'n': 10, 'p': 0.5})
    mb.add_latent_variable('d', pymc.Binomial, {'n': 10, 'p': 0.5})

    mb.define_experiment_params(['vdd', 'temp', 'time'], simultaneous_experiments=['one', 'two'],
                                samples_per_experiment={'all': 2})

    mb.add_dependent_variable('deg', happy_little_linear_degradation)
    mb.set_variable_observed('deg', variability=10)

    # This function is the inverse of the degradation mechanism with a set failure point
    def lifespan(a, b, c, d):
        fail_point, temp, vdd = -150, 300, 0.8
        # Time is the output, this means we may have to numerically estimate the inverse equation for many mechanisms :(
        time = 1000 * (fail_point / (-1 * d * ((a / 10) + 0.8) * (b * vdd) * (c * (temp / 100))))
        return time
    mb.add_lifespan_variable('fail_point', lifespan)

    tm = stratcona.TestDesignManager(mb)
    tm.set_experiment_conditions({'one': {'vdd': 0.8, 'temp': 300, 'time': 500},
                                  'two': {'vdd': 0.9, 'temp': 350, 'time': 500}})
    tm.examine('prior_predictive')
    plt.show()

    estimate = tm.estimate_reliability()
    print(estimate)

    ### Determine Best Experiment Step ###
    def exp_sampler():
        vdds_possible = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        temps_possible = [275, 300, 325, 350, 375, 400]
        times_possible = [100, 300, 500, 700, 900]
        as_dict = {}
        for exp in ['one', 'two']:
            as_dict[exp] = {'vdd': np.random.choice(vdds_possible),
                            'temp': np.random.choice(temps_possible),
                            'time': np.random.choice(times_possible)}
        return as_dict
    tm.determine_best_test(exp_sampler, (-400, 0))

    ### Simulate the Experiment Step ###

    ### Inference Step ###
    tm.set_experiment_conditions({'one': {'vdd': 0.8, 'temp': 300, 'time': 500},
                                  'two': {'vdd': 0.9, 'temp': 350, 'time': 500}})
    tm.infer_model({'one': {'deg': np.array([-40, -49])}, 'two': {'deg': np.array([-47, -79])}})
    new_estimate = tm.estimate_reliability()
    print(new_estimate)


if __name__ == '__main__':
    discrete_wearout_demo()
