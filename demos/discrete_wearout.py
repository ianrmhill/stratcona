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

    def param_reduction(a, b, c, temp, time):
        return -1 * (a + 1) * np.exp((b * temp) / 1000) * (time ** (c / 100))

    mb.add_latent_variable('a', pymc.Categorical, {'p': [0.0, 0.2, 0.3, 0.3, 0.1, 0.1]})
    mb.add_latent_variable('b', pymc.Categorical, {'p': [0.0, 0.2, 0.3, 0.3, 0.2, 0.0]})
    mb.add_latent_variable('c', pymc.Categorical, {'p': [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]})

    mb.define_experiment_params(['temp', 'time'], simultaneous_experiments=1, samples_per_experiment=10)

    mb.add_dependent_variable('deg', param_reduction)
    mb.set_variable_observed('deg', variability=0.001)

    # This function is the inverse of the degradation mechanism with a set failure point
    def lifespan(a, b, c):
        fail_point, temp = -5, 300
        # Time is the output, this means we may have to numerically estimate the inverse equation for many mechanisms :(
        time = (fail_point / (-1 * (a + 1) * np.exp((b * temp) / 1000))) ** (100 / c)
        return time
    mb.add_lifespan_variable('fail_point', lifespan)

    tm = stratcona.TestDesignManager(mb)
    #tm.examine('latents')
    #plt.show()

    tm.set_experiment_conditions({'exp1': {'temp': 300, 'time': 1000}})
    estimate = tm.estimate_reliability()
    print(estimate)
    #tm.determine_best_test()
    tm.infer_model({'deg': np.array([[-4], [-5], [-4], [-5], [-6], [-7], [-5], [-6], [-4], [-4]])})
    new_estimate = tm.estimate_reliability()
    print(new_estimate)


if __name__ == '__main__':
    discrete_wearout_demo()
