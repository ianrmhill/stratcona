# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import time as t
import numpy as np
import pytensor.tensor as pt
import pymc

import gerabaldi
from gerabaldi.models import *

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TEMP
from matplotlib import pyplot as plt

import stratcona


def threshold_demo():
    ### Define the model we will use to fit degradation ###
    mb = stratcona.ModelBuilder(mdl_name='Threshold Degradation')

    def threshold_degradation(a, c, vdd, temp, time):
        #return pt.where(pt.gt(vdd, a), b * (temp - (c * 100)) * (time / 1000), 0)
        return pt.where(pt.gt(vdd, a), (temp - (c * 100)) * (time / 1000), 0)

    mb.add_latent_variable('a', pymc.Normal, {'mu': 0.85, 'sigma': 0.1})
    #mb.add_latent_variable('b', pymc.Beta, {'alpha': 2.0, 'beta': 3.0})
    #mb.add_latent_variable('b', pymc.Normal, {'mu': 0.3, 'sigma': 0.1})
    mb.add_latent_variable('c', pymc.Normal, {'mu': 1.5, 'sigma': 0.1})

    mb.define_experiment_params(['vdd', 'temp', 'time'], simultaneous_experiments=['single'],
                                samples_per_experiment={'all': 5})

    mb.add_dependent_variable('deg', threshold_degradation)
    mb.set_variable_observed('deg', variability=2)

    # This function is the inverse of the degradation mechanism with a set failure point
    mb.gen_lifespan_variable('fail_point', fail_bounds={'deg': 40}, field_use_conds={'temp': 300, 'vdd': 0.85})

    ### Compile the model for use ###
    tm = stratcona.TestDesignManager(mb)

    tm.set_experiment_conditions({'single': {'vdd': 0.85, 'temp': 300, 'time': 500}})
    tm.examine('prior_predictive')

    start_time = t.time()
    #estimate = tm.estimate_reliability(num_samples=3000)
    print(f"Lifespan prediction time: {t.time() - start_time} seconds")
    #print(f"Estimated product lifespan: {estimate} hours")

    #tm.examine('lifespan')
    #plt.show()

    ### Determine Best Experiment ###
    exp_sampler = stratcona.assistants.iterator.iter_sampler([
        {'vdd': 0.55, 'temp': 300, 'time': 500}, {'vdd': 0.60, 'temp': 300, 'time': 500},
        {'vdd': 0.65, 'temp': 300, 'time': 500}, {'vdd': 0.70, 'temp': 300, 'time': 500},
        {'vdd': 0.75, 'temp': 300, 'time': 500}, {'vdd': 0.80, 'temp': 300, 'time': 500},
        {'vdd': 0.85, 'temp': 300, 'time': 500}, {'vdd': 0.90, 'temp': 300, 'time': 500},
        {'vdd': 0.95, 'temp': 300, 'time': 500}, {'vdd': 1.00, 'temp': 300, 'time': 500},
        {'vdd': 1.05, 'temp': 300, 'time': 500}, {'vdd': 1.10, 'temp': 300, 'time': 500},
        {'vdd': 1.15, 'temp': 300, 'time': 500}, {'vdd': 1.20, 'temp': 300, 'time': 500},
        {'vdd': 1.25, 'temp': 300, 'time': 500}, {'vdd': 1.30, 'temp': 300, 'time': 500}])

    start_time = t.time()
    #tm.determine_best_test(exp_sampler, (-5, 120), num_tests_to_eval=16, num_obs_samples_per_test=300, num_ltnt_samples_per_test=300)
    print(f"Test EIG estimation time: {t.time() - start_time} seconds")

    ### Simulate the Experiment ###
    to_meas = MeasSpec({'deg': 5}, {'temp': 300, 'vdd': 0.85}, 'Measure Ten')
    # We will run two tests, one with high EIG, one with poor EIG
    best_strs = StrsSpec({'temp': 300, 'vdd': 0.84}, 500, 'Best Vdd')
    poor_strs = StrsSpec({'temp': 300, 'vdd': 1.15}, 500, 'Poor Vdd')
    best_test = TestSpec([best_strs, to_meas], num_chps=1, num_lots=1, name='Best Test')
    poor_test = TestSpec([poor_strs, to_meas], num_chps=1, num_lots=1, name='Poor Test')

    test_env = PhysTestEnv(env_vrtns={
        'temp': EnvVrtnMdl(dev_vrtn_mdl=Normal(0, 0.1), chp_vrtn_mdl=Normal(0, 0.4)),
        'vdd': EnvVrtnMdl(dev_vrtn_mdl=Normal(0, 0.003))
    })

    # The actual model is not of the same form as the model used for reliability prediction (here a power term is added)
    #def threshold_degradation(a, b, c, d, vdd, temp, time):
    #    return np.where(np.greater(vdd, a), (b * (temp - (c * 100)) * (time / 1000)) ** d, 0)

    def threshold_degradation(a, c, vdd, temp, time):
        #d = np.where(np.greater(vdd, a), b * (temp - (c * 100)) * (time / 1000), 0)
        d = np.where(np.greater(vdd, a), (temp - (c * 100)) * (time / 1000), 0)
        return d

    dev_mdl = DeviceMdl({'deg': DegPrmMdl(DegMechMdl(
        threshold_degradation, mdl_name='Threshold',
        a=LatentVar(deter_val=0.84),
        #b=LatentVar(deter_val=0.39), c=LatentVar(deter_val=1.48), d=LatentVar(deter_val=0.96)))})
        c=LatentVar(deter_val=1.48)))})

    start_time = t.time()
    best_test_rslts = gerabaldi.simulate(best_test, dev_mdl, test_env)
    poor_test_rslts = gerabaldi.simulate(poor_test, dev_mdl, test_env)
    print(f"Simulation time: {t.time() - start_time} seconds")

    # Extract the 10 raw values from the test results
    best_vals = best_test_rslts.measurements['measured'].to_numpy()
    poor_vals = poor_test_rslts.measurements['measured'].to_numpy()
    print(f"Best test simulation measured: {best_vals}.")
    print(f"Poor test simulation measured: {poor_vals}.")

    ### Inference Step for Each Test ###
    priors = tm.get_priors(for_user=True)
    tm.set_experiment_conditions({'single': {'vdd': 0.84, 'temp': 300, 'time': 500}})
    # FIXME: The inference model has all 5 observations sharing latent variable samples, thus either all five must meet
    #        the threshold or none, making inference incorrect
    tm.infer_model({'single': {'deg': best_vals}})

    # Reset model for second inference
    tm.set_priors(priors)
    tm.set_experiment_conditions({'single': {'vdd': 1.15, 'temp': 300, 'time': 500}})
    tm.infer_model({'single': {'deg': poor_vals}})

    new_estimate = tm.estimate_reliability()
    print(new_estimate)


if __name__ == '__main__':
    threshold_demo()
