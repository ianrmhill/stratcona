# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import time as t
import numpy as np
import pytensor.tensor as pt
import pymc
from functools import partial

import gerabaldi
from gerabaldi.models import *

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TEMP
from matplotlib import pyplot as plt

import stratcona

BOLTZ_EV = 8.617e-5


def idfbcamp_demo():
    mb = stratcona.ModelBuilder(mdl_name='IP Block Showcase')

    num_devices = 3
    mb.define_experiment_params(['time', 'vdd', 'temp'], simultaneous_experiments=['t1'],
                                samples_per_observation={'em_sensor': 8, 'ring_osc_freq': 5})

    # Model provided in JEDEC's JEP122H as general empirical NBTI degradation model, equation 5.3.1
    def bti_vth_shift_empirical(time, vdd, temp, a0, e_aa, bti_alpha, bti_n):
        return 1000 * (a0 * 0.001) * np.exp((e_aa * 0.01) / (BOLTZ_EV * temp)) * (vdd ** bti_alpha) * (time ** bti_n)

    mb.add_latent_variable('a0', pymc.Normal, {'mu': 6, 'sigma': 0.5})
    mb.add_latent_variable('e_aa', pymc.Normal, {'mu': -5, 'sigma': 0.3})
    mb.add_latent_variable('bti_alpha', pymc.Normal, {'mu': 9.5, 'sigma': 0.5})
    mb.add_latent_variable('bti_n', pymc.Normal, {'mu': 0.4, 'sigma': 0.1})

    # HCI model from Takeda and Suzuki, DOI: https://doi.org/10.1109/EDL.1983.25667
    # Some basic negative temperature dependence term added to enrich the demo
    def hci_vth_shift_empirical(time, vdd, temp, h0, hci_alpha, hci_beta, hci_n):
        return 1000 * (h0 * 10) * np.exp(-hci_alpha / vdd) * (temp ** -hci_beta) * (time ** hci_n)

    mb.add_latent_variable('h0', pymc.Normal, {'mu': 0.5, 'sigma': 0.1})
    mb.add_latent_variable('hci_alpha', pymc.Normal, {'mu': 7.2, 'sigma': 0.2})
    mb.add_latent_variable('hci_beta', pymc.Normal, {'mu': 1.1, 'sigma': 0.1})
    mb.add_latent_variable('hci_n', pymc.Normal, {'mu': 0.62, 'sigma': 0.1})

    # Express wire current density as a function of the number of transistors and the voltage applied
    def j_n(n_fins, vdd, vth_avg, v_c, i_scale):
        return pt.where(pt.gt(vdd, vth_avg), n_fins * (i_scale * 10) * ((vdd - vth_avg) ** v_c), 0.0)

    # TODO: vth_base could be inferenced prior to accelerated wearout testing, would want to be excluded from EIG and
    #       inference, parameterization is fixed (may cause the model to have implicit likelihood!). Could also just
    #       make it a constant like BOLTZ_EV. Other params may be like this too! Only the wear-out related ones should
    #       stay.
    mb.add_latent_variable('vth_avg', pymc.Normal, {'mu': 0.315, 'sigma': 0.03})
    mb.add_latent_variable('v_c', pymc.Normal, {'mu': 1.3, 'sigma': 0.1})
    mb.add_latent_variable('i_scale', pymc.Normal, {'mu': 0.8, 'sigma': 0.05}) # In mA

    # The classic model for electromigration failure estimates, DOI: 10.1109/T-ED.1969.16754
    # Added variance to try and characterize the PDF of failure times, not just the mean
    WIRE_AREA = 1.024 * 1000 * 1000 # In nm^2
    def em_blacks_equation(j_n_sensor, temp, variance, em_n, em_e_aa):
        return variance * (WIRE_AREA / (j_n_sensor ** em_n)) * np.exp((em_e_aa * 0.01) / (BOLTZ_EV * temp))

    mb.add_latent_variable('variance', pymc.Normal, {'mu': 1, 'sigma': 0.1})
    mb.add_latent_variable('em_n', pymc.Normal, {'mu': 1.8, 'sigma': 0.04})
    mb.add_latent_variable('em_e_aa', pymc.Normal, {'mu': 2, 'sigma': 0.2})

    # Now correspond the degradation mechanisms to the output values
    mb.add_dependent_variable('bti_shift', bti_vth_shift_empirical)
    mb.add_dependent_variable('hci_shift', hci_vth_shift_empirical)
    mb.add_dependent_variable('j_n_sensor', partial(j_n, n_fins=np.array([3, 6, 12, 24, 48, 96, 192, 384])))
    mb.add_dependent_variable('em_sensor', em_blacks_equation)

    mb.set_variable_observed('em_sensor', variability=1)

    # Following the RO inverter propagation delay analysis from MIT lecture: http://web.mit.edu/6.012/www/SP07-L13.pdf
    # The approximate per-stage delay is (1/2 * Q_L) / I_D for NMOS and PMOS with slightly different I_D, take their
    # average for typical delay. I_D is based on saturation mode as transistors haven't fully turned on yet, Q_L is
    # VDD * C_L (interconnect parasitics + next stage gate capacitance)
    def ring_osc_stage_delay(vdd, bti_shift, hci_shift, c_l, k_avg, vth_avg):
        # half_q_l = 0.5 * vdd * c_l
        # t_n = half_q_l / (k_n * ((vdd - vth_n) ** 2))
        # t_p = half_q_l / (k_p * ((vdd - abs(vth_p)) ** 2))
        # t_avg = 0.5 * (t_n + t_p)
        return (0.5 * vdd * (c_l * 1e-9)) / (k_avg * (vdd - (vth_avg + ((bti_shift + hci_shift) * 0.001)) ** 2))

    mb.add_latent_variable('c_l', pymc.Normal, {'mu': 0.3, 'sigma': 0.02}) # In pF
    mb.add_latent_variable('k_avg', pymc.Normal, {'mu': 2.4, 'sigma': 0.2})

    NUM_STAGES = 51
    def ring_osc_freq(ro_stage_delay):
        # 0.5 as each stage must toggle twice per period, once high->low, once low->high
        return 1 / (0.5 * NUM_STAGES * ro_stage_delay)

    mb.add_dependent_variable('ro_stage_delay', ring_osc_stage_delay)
    mb.add_dependent_variable('ring_osc_freq', ring_osc_freq)

    mb.set_variable_observed('ring_osc_freq', variability=1000)

    # This function is the inverse of the degradation mechanism with a set failure point
    #mb.gen_lifespan_variable('em_failure', fail_bounds={'em_sensor': 239})

    tm = stratcona.TestDesignManager(mb)
    tm.set_experiment_conditions({'t1': {'time': 10, 'vdd': 0.85, 'temp': 350}})
    tm.examine('prior_predictive')
    #estimate = tm.estimate_reliability(num_samples=3000)
    #print(f"Estimated product lifespan: {estimate} hours")
    plt.show()

    exp_sampler = stratcona.assistants.iterator.iter_sampler([{'t1': {'time': 10, 'samples': 6}}])

    start_time = t.time()
    tm.determine_best_test(exp_sampler, (-1, 1), num_tests_to_eval=1,
                           num_obs_samples_per_test=200, num_ltnt_samples_per_test=200)
    print(f"Test EIG estimation time: {t.time() - start_time} seconds")


if __name__ == '__main__':
    idfbcamp_demo()
