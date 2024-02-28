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


def threshold_demo():
    mb = stratcona.ModelBuilder(mdl_name='Threshold Degradation')

    def sudden_failures(d, temp, time):
        # Placing time in a random choice should mask d's impact on observations unless time is increased
        # Temperature should minimally affect the test quality unless it's 0
        p = d * (temp / 600) * np.random.choice(2, 1, p=[1 - (time / 2000), (time / 2000)])
        return np.random.choice(2, 1, p=[1 - p, p])

    def threshold_degradation(a, b, c, vdd, time):
        return pt.tensor.where(pt.tensor.gt(vdd, a), (b * 10) * (vdd - c) * (time / 1000), 0)

    mb.add_latent_variable('a', pymc.Normal, {'mu': 0.85, 'sigma': 0.03})
    mb.add_latent_variable('b', pymc.Beta, {'alpha': 2.0, 'beta': 3.0})
    mb.add_latent_variable('c', pymc.Normal, {'mu': 0.75, 'sigma': 0.05})
    #mb.add_latent_variable('d', pymc.Normal, {'mu': 0.5, 'sigma': 0.05})

    mb.define_experiment_params(['vdd', 'temp', 'time'], simultaneous_experiments=['single'],
                                samples_per_experiment={'all': 3})

    mb.add_dependent_variable('deg', threshold_degradation)
    mb.set_variable_observed('deg', variability=0.02)

    #mb.add_dependent_variable('fail', sudden_failures)
    #mb.set_variable_observed('fail', variability=0.05)

    # This function is the inverse of the degradation mechanism with a set failure point
    mb.gen_lifespan_variable('fail_point', fail_bounds={'deg': 1}, field_use_conds={'temp': 300, 'vdd': 0.9})

    tm = stratcona.TestDesignManager(mb)
    tm.set_experiment_conditions({'single': {'vdd': 0.87, 'temp': 300, 'time': 500}})
    tm.examine('prior_predictive')
    plt.show()

    estimate = tm.estimate_reliability(num_samples=3)
    tm.examine('lifespan')
    print(f"Estimated product lifespan: {estimate} hours")

    ### Determine Best Experiment Step ###
    def exp_sampler():
        vdds_possible = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        temps_possible = [275, 300, 325, 400]
        times_possible = [100, 300, 500, 700, 900]
        as_dict = {}
        for exp in ['single']:
            as_dict[exp] = {'vdd': np.random.choice(vdds_possible),
                            'temp': np.random.choice(temps_possible),
                            'time': np.random.choice(times_possible)}
        return as_dict
    #tm.determine_best_test(exp_sampler, (0, 1))

    ### Simulate the Experiment Step ###

    ### Inference Step ###
    tm.set_experiment_conditions({'single': {'vdd': 0.85, 'temp': 300, 'time': 500}})
    #tm.infer_model({'single': {'deg': np.array([0.3, 0.4, 0.35])}})
    new_estimate = tm.estimate_reliability()
    print(new_estimate)


if __name__ == '__main__':
    threshold_demo()
