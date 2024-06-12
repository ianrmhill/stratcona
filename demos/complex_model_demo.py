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

BOLTZMANN_CONST_EV = 8.617e-5


def idfbcamp_demo():
    mb = stratcona.ModelBuilder(mdl_name='IP Block Showcase')

    num_devices = 3
    mb.define_experiment_params(['time'], simultaneous_experiments=['t1'],
                                samples_per_observation={'all': num_devices})

    # Model provided in JEDEC's JEP122H as general empirical NBTI degradation model, equation 5.3.1
    def bti_vth_shift_empirical(time, vdd, temp, a0, e_aa, bti_alpha, n):
        return a0 * np.exp(e_aa / (BOLTZMANN_CONST_EV * temp)) * (vdd ** bti_alpha) * (time ** n)

    # HCI model from Takeda and Suzuki, DOI: https://doi.org/10.1109/EDL.1983.25667
    # Some basic negative temperature dependence term added to enrich the demo
    def hci_vth_shift_empirical(time, vdd, temp, h0, n, t_0, beta, alpha):
        return h0 * np.exp(-alpha / vdd) * (t_0 * (temp ** -beta)) * (time ** n)

    # TODO: Use proper EM model
    def em_fail_empirical(time, vdd, temp, em_c):
        return em_c * temp * vdd * time

    mb.add_latent_variable('ba_mean', pymc.Normal, {'mu': 0.7, 'sigma': 0.1})
    mb.add_latent_variable('ba_dev', pymc.Normal, {'mu': 0, 'sigma': 0.1})
    def bti_alpha(ba_mean, ba_dev):
        return ba_mean + ba_dev
    mb.add_dependent_variable('bti_alpha', bti_alpha)

    mb.add_latent_variable('a0_mean', pymc.Normal, {'mu': 0.7, 'sigma': 0.1})
    mb.add_latent_variable('a0_dev', pymc.Normal, {'mu': 0, 'sigma': 0.1})
    def a0(a0_mean, a0_dev):
        return a0_mean + a0_dev
    mb.add_dependent_variable('a0', a0)


    mb.add_dependent_variable('bti_shift', bti_vth_shift_empirical)
    mb.add_dependent_variable('hci_shift', hci_vth_shift_empirical)

    mb.set_variable_observed('deg', variability=0.1)
    # TODO: Observed variable that is bernoulli with probability output to represent EM line failures observed

    # This function is the inverse of the degradation mechanism with a set failure point
    mb.gen_lifespan_variable('fail', fail_bounds={'deg': -23})

    tm = stratcona.TestDesignManager(mb)
    tm.set_experiment_conditions({'t1': {'time': 10}})
    tm.examine('prior_predictive')
    estimate = tm.estimate_reliability(num_samples=3000)
    print(f"Estimated product lifespan: {estimate} hours")
    plt.show()

    exp_sampler = stratcona.assistants.iterator.iter_sampler([{'t1': {'time': 10, 'samples': 6}}])

    start_time = t.time()
    tm.determine_best_test(exp_sampler, (-1, 1), num_tests_to_eval=1,
                           num_obs_samples_per_test=200, num_ltnt_samples_per_test=200)
    print(f"Test EIG estimation time: {t.time() - start_time} seconds")


if __name__ == '__main__':
    idfbcamp_demo()
