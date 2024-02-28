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


def lifespan_demo():
    mb = stratcona.ModelBuilder(mdl_name='Lifespan Estimation Functionality')

    def linear_slope(a, time):
        return a * time

    mb.add_latent_variable('a', pymc.Normal, {'mu': 0.2, 'sigma': 0.04})
    mb.define_experiment_params(['time'])
    mb.add_dependent_variable('increase', linear_slope)
    mb.set_variable_observed('increase', variability=1)

    mb.gen_lifespan_variable('failed', fail_bounds={'increase': 100})

    tm = stratcona.TestDesignManager(mb)
    tm.set_experiment_conditions({'time': 50})
    tm.examine('prior_predictive')

    estimate = tm.estimate_reliability(num_samples=3000)
    tm.examine('lifespan')
    plt.show()
    print(f"Estimated product lifespan: {estimate} hours")

    ### Simulate the Experiment Step ###

    ### Inference Step ###
    #tm.set_experiment_conditions({'time': 50})
    #tm.infer_model({'increase': np.array([56])})
    #new_estimate = tm.estimate_reliability()
    #print(new_estimate)


if __name__ == '__main__':
    lifespan_demo()
