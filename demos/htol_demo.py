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


def htol_demo():
    ### Define some constants ###
    num_devices = 5
    boltz_ev = 8.617e-5
    # 125C in Kelvin
    htol_temp = 398.15

    ########################################################################
    ### 1. Define the predictive wear-out model to infer                 ###
    ########################################################################
    ### Define the model we will use to fit degradation ###
    mb = stratcona.ModelBuilder(mdl_name='HTOL Test')

    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1
    def bti_vth_shift_empirical(a0, e_aa, temp, vdd, alpha, time, n):
        return 1000 * (a0 * 0.001) * pt.exp((e_aa * 0.01) / (boltz_ev * temp)) * (vdd ** alpha) * (time ** n)

    mb.add_latent_variable('a0', pymc.Normal, {'mu': 6, 'sigma': 0.5})
    mb.add_latent_variable('e_aa', pymc.Normal, {'mu': -5, 'sigma': 0.3})
    mb.add_latent_variable('alpha', pymc.Normal, {'mu': 9.5, 'sigma': 0.5})
    mb.add_latent_variable('n', pymc.Normal, {'mu': 0.4, 'sigma': 0.1})

    mb.define_experiment_params(['vdd', 'temp', 'time'], simultaneous_experiments=['lot1', 'lot2', 'lot3'],
                                samples_per_observation={'all': num_devices})

    mb.add_dependent_variable('delta_vth', bti_vth_shift_empirical)
    mb.set_variable_observed('delta_vth', variability=25)

    # This function is the inverse of the degradation mechanism with a set failure point
    mb.gen_lifespan_variable('fail_point', fail_bounds={'delta_vth': 0.3}, field_use_conds={'temp': htol_temp, 'vdd': 1.15})

    ########################################################################
    ### 2. Compile the model for use                                     ###
    ########################################################################
    tm = stratcona.TestDesignManager(mb)

    tm.set_experiment_conditions({'lot1': {'vdd': 1.05, 'temp': htol_temp, 'time': 1000},
                                  'lot2': {'vdd': 1.15, 'temp': htol_temp, 'time': 1000},
                                  'lot3': {'vdd': 1.15, 'temp': htol_temp - 15, 'time': 1000}})
    #tm.examine('prior_predictive')

    #start_time = t.time()
    #estimate = tm.estimate_reliability(num_samples=3000)
    #print(f"Estimated product lifespan: {estimate} hours")
    #print(f"Lifespan prediction time: {t.time() - start_time} seconds")

    #plt.show()

    ########################################################################
    ### 3. Determine the best experiment to conduct                      ###
    ########################################################################
    exp_sampler = stratcona.assistants.iterator.iter_sampler([
        {'lot1': {'vdd': 1.05, 'temp': htol_temp, 'time': 1000, 'samples': 5},
         'lot2': {'vdd': 1.15, 'temp': htol_temp, 'time': 1000, 'samples': 5},
         'lot3': {'vdd': 1.15, 'temp': htol_temp - 15, 'time': 1000, 'samples': 5}}])

    start_time = t.time()
    tm.determine_best_test(exp_sampler, (-0.2, 1.2), num_tests_to_eval=1,
                           num_obs_samples_per_test=200, num_ltnt_samples_per_test=200)
    print(f"Test EIG estimation time: {t.time() - start_time} seconds")

    return
    ########################################################################
    ### 4. Simulate the Experiment                                       ###
    ########################################################################
    to_meas = MeasSpec({'deg': num_devices}, {'temp': 300, 'vdd': 0.85}, 'Measure Ten')
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
    tm.set_experiment_conditions({'htol': {'vdd': 0.84, 'temp': 300, 'time': 500}})
    tm.infer_model({'htol': {'delta_vth': best_vals}})

    # Reset model for second inference
    tm.set_priors(priors)
    tm.set_experiment_conditions({'htol': {'vdd': 1.15, 'temp': 300, 'time': 500}})
    tm.infer_model({'htol': {'delta_vth': poor_vals}})

    new_estimate = tm.estimate_reliability()
    print(new_estimate)


if __name__ == '__main__':
    htol_demo()
