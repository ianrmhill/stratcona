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

    mb.define_experiment_params(
        ['time', 'vdd', 'temp'], simultaneous_experiments=['t1'],
        samples_per_observation={'em_sensor': 8, 'reg_ro_freq': 3,
                                 'nbti_ro_freq': 5, 'pbti_ro_freq': 5, 'phci_ro_freq': 5, 'nhci_ro_freq': 5,
                                 'nbti_vamp_out': 5, 'pbti_vamp_out': 5, 'phci_vamp_out': 5, 'nhci_vamp_out': 5})

    # Model provided in JEDEC's JEP122H as general empirical NBTI degradation model, equation 5.3.1
    def nbti_vth_shift_empirical(time, vdd, temp, nbti_a0, nbti_eaa, nbti_alpha, nbti_n, v_adjust, t_adjust):
        return (nbti_a0 * 0.001) * np.exp((nbti_eaa * 0.01) / (BOLTZ_EV * temp)) *\
               ((vdd * v_adjust) ** nbti_alpha) * ((time * t_adjust) ** nbti_n)
    mb.add_latent_variable('nbti_a0', pymc.Normal, {'mu': 6, 'sigma': 0.5})
    mb.add_latent_variable('nbti_eaa', pymc.Normal, {'mu': -5, 'sigma': 0.3})
    mb.add_latent_variable('nbti_alpha', pymc.Normal, {'mu': 9.5, 'sigma': 0.5})
    mb.add_latent_variable('nbti_n', pymc.Normal, {'mu': 0.4, 'sigma': 0.1})
    def pbti_vth_shift_empirical(time, vdd, temp, pbti_a0, pbti_eaa, pbti_alpha, pbti_n, v_adjust, t_adjust):
        return (pbti_a0 * 0.001) * np.exp((pbti_eaa * 0.01) / (BOLTZ_EV * temp)) * \
               ((vdd * v_adjust) ** pbti_alpha) * ((time * t_adjust) ** pbti_n)
    mb.add_latent_variable('pbti_a0', pymc.Normal, {'mu': 6, 'sigma': 0.5})
    mb.add_latent_variable('pbti_eaa', pymc.Normal, {'mu': -5, 'sigma': 0.3})
    mb.add_latent_variable('pbti_alpha', pymc.Normal, {'mu': 9.5, 'sigma': 0.5})
    mb.add_latent_variable('pbti_n', pymc.Normal, {'mu': 0.4, 'sigma': 0.1})

    # HCI model from Takeda and Suzuki, DOI: https://doi.org/10.1109/EDL.1983.25667
    # Some basic negative temperature dependence term added to enrich the demo
    def phci_vth_shift_empirical(time, vdd, temp, phci_h0, phci_alpha, phci_beta, phci_n, v_adjust, t_adjust):
        return (phci_h0 * 10) * np.exp(-phci_alpha / (vdd * v_adjust)) *\
               (temp ** -phci_beta) * ((time * t_adjust) ** phci_n)
    mb.add_latent_variable('phci_h0', pymc.Normal, {'mu': 0.5, 'sigma': 0.1})
    mb.add_latent_variable('phci_alpha', pymc.Normal, {'mu': 7.2, 'sigma': 0.2})
    mb.add_latent_variable('phci_beta', pymc.Normal, {'mu': 1.1, 'sigma': 0.1})
    mb.add_latent_variable('phci_n', pymc.Normal, {'mu': 0.62, 'sigma': 0.1})
    def nhci_vth_shift_empirical(time, vdd, temp, nhci_h0, nhci_alpha, nhci_beta, nhci_n, v_adjust, t_adjust):
        return (nhci_h0 * 10) * np.exp(-nhci_alpha / (vdd * v_adjust)) *\
               (temp ** -nhci_beta) * ((time * t_adjust) ** nhci_n)
    mb.add_latent_variable('nhci_h0', pymc.Normal, {'mu': 0.5, 'sigma': 0.1})
    mb.add_latent_variable('nhci_alpha', pymc.Normal, {'mu': 7.2, 'sigma': 0.2})
    mb.add_latent_variable('nhci_beta', pymc.Normal, {'mu': 1.1, 'sigma': 0.1})
    mb.add_latent_variable('nhci_n', pymc.Normal, {'mu': 0.62, 'sigma': 0.1})

    ### STANDARD RING OSCILLATOR ###
    mb.add_dependent_variable('nbti_shift_reg_ro', partial(nbti_vth_shift_empirical, v_adjust=1.0, t_adjust=0.45))
    mb.add_dependent_variable('pbti_shift_reg_ro', partial(pbti_vth_shift_empirical, v_adjust=1.0, t_adjust=0.45))
    mb.add_dependent_variable('phci_shift_reg_ro', partial(phci_vth_shift_empirical, v_adjust=0.85, t_adjust=0.05))
    mb.add_dependent_variable('nhci_shift_reg_ro', partial(nhci_vth_shift_empirical, v_adjust=0.85, t_adjust=0.05))
    vtn_typ, vtp_typ = 0.315, 0.325
    def reg_ro_vth_avg(nbti_shift_reg_ro, pbti_shift_reg_ro, phci_shift_reg_ro, nhci_shift_reg_ro):
        return 0.5 * ((vtn_typ - pbti_shift_reg_ro - nhci_shift_reg_ro) + (vtp_typ - nbti_shift_reg_ro - phci_shift_reg_ro))
    mb.add_dependent_variable('reg_ro_vth_avg', reg_ro_vth_avg)
    # Following the RO inverter propagation delay analysis from MIT lecture: http://web.mit.edu/6.012/www/SP07-L13.pdf
    # The approximate per-stage delay is (1/2 * Q_L) / I_D for NMOS and PMOS with slightly different I_D, take their
    # average for typical delay. I_D is based on saturation mode as transistors haven't fully turned on yet, Q_L is
    # VDD * C_L (interconnect parasitics + next stage gate capacitance)
    vdd_meas, reg_ro_cl, k_avg = 0.8, 0.3, 13.4 # C_l in pF
    def reg_ro_stage_delay(reg_ro_vth_avg):
        # half_q_l = 0.5 * vdd * c_l
        # t_n = half_q_l / (k_n * ((vdd - vth_n) ** 2))
        # t_p = half_q_l / (k_p * ((vdd - abs(vth_p)) ** 2))
        # t_avg = 0.5 * (t_n + t_p)
        return (0.5 * vdd_meas * (reg_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - reg_ro_vth_avg) ** 2))
    mb.add_dependent_variable('reg_ro_stage_delay', reg_ro_stage_delay)
    num_stages = 51
    def reg_ro_freq(reg_ro_stage_delay):
        # 0.5 as each stage must toggle twice per period, once high->low, once low->high
        return 1 / (0.5 * num_stages * reg_ro_stage_delay)
    mb.add_dependent_variable('reg_ro_freq', reg_ro_freq)
    mb.set_variable_observed('reg_ro_freq', variability=1000)

    ### ISOLATION SENSOR VTH SHIFTS ###
    mb.add_dependent_variable('nbti_shift_full', partial(nbti_vth_shift_empirical, v_adjust=1.0, t_adjust=1.0))
    mb.add_dependent_variable('nbti_shift_hci_stress', partial(nbti_vth_shift_empirical, v_adjust=0.85, t_adjust=1.0))
    mb.add_dependent_variable('pbti_shift_full', partial(pbti_vth_shift_empirical, v_adjust=1.0, t_adjust=1.0))
    mb.add_dependent_variable('pbti_shift_hci_stress', partial(pbti_vth_shift_empirical, v_adjust=0.85, t_adjust=1.0))
    mb.add_dependent_variable('phci_shift_full', partial(phci_vth_shift_empirical, v_adjust=1.0, t_adjust=1.0))
    mb.add_dependent_variable('nhci_shift_full', partial(nhci_vth_shift_empirical, v_adjust=1.0, t_adjust=1.0))
    def nbti_vth(nbti_shift_full):
        return vtp_typ - nbti_shift_full
    mb.add_dependent_variable('nbti_vth', nbti_vth)
    def pbti_vth(pbti_shift_full):
        return vtn_typ - pbti_shift_full
    mb.add_dependent_variable('pbti_vth', pbti_vth)
    def phci_vth(nbti_shift_hci_stress, phci_shift_full):
        return vtp_typ - nbti_shift_hci_stress - phci_shift_full
    mb.add_dependent_variable('phci_vth', phci_vth)
    def nhci_vth(pbti_shift_hci_stress, nhci_shift_full):
        return vtn_typ - pbti_shift_hci_stress - nhci_shift_full
    mb.add_dependent_variable('nhci_vth', nhci_vth)

    ### SPECIAL RING OSCILLATOR SENSORS ###
    bti_ro_cl, hci_ro_cl = 0.6, 0.65
    def nbti_ro_freq(nbti_vth):
        nbti_d = (0.5 * vdd_meas * (bti_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (nbti_vth + vtn_typ))) ** 2))
        return 1 / (0.5 * num_stages * nbti_d)
    mb.add_dependent_variable('nbti_ro_freq', nbti_ro_freq)
    mb.set_variable_observed('nbti_ro_freq', variability=1000)
    def pbti_ro_freq(pbti_vth):
        pbti_d = (0.5 * vdd_meas * (bti_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (pbti_vth + vtp_typ))) ** 2))
        return 1 / (0.5 * num_stages * pbti_d)
    mb.add_dependent_variable('pbti_ro_freq', pbti_ro_freq)
    mb.set_variable_observed('pbti_ro_freq', variability=1000)
    def phci_ro_freq(phci_vth):
        phci_d = (0.5 * vdd_meas * (hci_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (phci_vth + vtn_typ))) ** 2))
        return 1 / (0.5 * num_stages * phci_d)
    mb.add_dependent_variable('phci_ro_freq', phci_ro_freq)
    mb.set_variable_observed('phci_ro_freq', variability=1000)
    def nhci_ro_freq(nhci_vth):
        nhci_d = (0.5 * vdd_meas * (hci_ro_cl * 1e-9)) / (k_avg * ((vdd_meas - (0.5 * (nhci_vth + vtp_typ))) ** 2))
        return 1 / (0.5 * num_stages * nhci_d)
    mb.add_dependent_variable('nhci_ro_freq', nhci_ro_freq)
    mb.set_variable_observed('nhci_ro_freq', variability=1000)

    ### OFFSET VOLTAGE SENSOR ###
    namp_gain, namp_vbase, pamp_gain, pamp_vbase, gm, ro = 5.4, 0.614, 5.2, 0.410, 0.042, 400
    def nbti_vamp_out(nbti_vth):
        return namp_vbase - (namp_gain * (-0.5 * (nbti_vth - vtp_typ) * gm * (0.5 * ro)))
    mb.add_dependent_variable('nbti_vamp_out', nbti_vamp_out)
    mb.set_variable_observed('nbti_vamp_out', variability=0.005)
    def pbti_vamp_out(pbti_vth):
        return pamp_vbase - (pamp_gain * (-0.5 * (pbti_vth - vtn_typ) * gm * (0.5 * ro)))
    mb.add_dependent_variable('pbti_vamp_out', pbti_vamp_out)
    mb.set_variable_observed('pbti_vamp_out', variability=0.005)
    def phci_vamp_out(phci_vth):
        return namp_vbase - (namp_gain * (-0.5 * (phci_vth - vtp_typ) * gm * (0.5 * ro)))
    mb.add_dependent_variable('phci_vamp_out', phci_vamp_out)
    mb.set_variable_observed('phci_vamp_out', variability=0.005)
    def nhci_vamp_out(nhci_vth):
        return pamp_vbase - (pamp_gain * (-0.5 * (nhci_vth - vtn_typ) * gm * (0.5 * ro)))
    mb.add_dependent_variable('nhci_vamp_out', nhci_vamp_out)
    mb.set_variable_observed('nhci_vamp_out', variability=0.005)

    ### ELECTROMIGRATION SENSOR ###
    # Express wire current density as a function of the number of transistors and the voltage applied
    vth_typ, i_base = 0.320, 0.8
    def j_n(n_fins, vdd):
        return pt.where(pt.gt(vdd, vth_typ), n_fins * i_base * ((vdd - vth_typ) ** 2), 0.0)
    mb.add_dependent_variable('jn_sensor', partial(j_n, n_fins=np.array([3, 6, 12, 24, 48, 96, 192, 384])))
    # The classic model for electromigration failure estimates, DOI: 10.1109/T-ED.1969.16754
    # Added variance to try and characterize the PDF of failure times, not just the mean
    wire_area = 1.024 * 1000 * 1000 # In nm^2
    def em_blacks_equation(jn_sensor, temp, variance, em_n, em_eaa):
        return variance * (wire_area / (jn_sensor ** em_n)) * np.exp((em_eaa * 0.01) / (BOLTZ_EV * temp))
    mb.add_latent_variable('variance', pymc.Normal, {'mu': 1, 'sigma': 0.1})
    mb.add_latent_variable('em_n', pymc.Normal, {'mu': 1.8, 'sigma': 0.04})
    mb.add_latent_variable('em_eaa', pymc.Normal, {'mu': 2, 'sigma': 0.2})
    # Now correspond the degradation mechanisms to the output values
    mb.add_dependent_variable('em_sensor', em_blacks_equation)
    # Add variability to electromigration sensor line failure times corresponding to 7% of the average time
    em_var = 0.07
    def em_variability(em_sensor):
        return em_var * em_sensor
    mb.add_dependent_variable('em_variability', em_variability)
    mb.set_variable_observed('em_sensor', variability='em_variability')

    # This function is the inverse of the degradation mechanism with a set failure point
    #mb.gen_lifespan_variable('em_failure', fail_bounds={'em_sensor': 239})

    tm = stratcona.TestDesignManager(mb)
    tm.set_experiment_conditions({'t1': {'time': 10, 'vdd': 0.85, 'temp': 350}})
    tm.examine('prior_predictive')
    #estimate = tm.estimate_reliability(num_samples=3000)
    #print(f"Estimated product lifespan: {estimate} hours")
    plt.show()

    exp_sampler = stratcona.assistants.iterator.iter_sampler([{'t1': {'time': 10, 'vdd': 0.85, 'temp': 350}}])

    start_time = t.time()
    tm.determine_best_test(exp_sampler, (-1, 1), num_tests_to_eval=1,
                           num_obs_samples_per_test=200, num_ltnt_samples_per_test=200)
    print(f"Test EIG estimation time: {t.time() - start_time} seconds")


if __name__ == '__main__':
    idfbcamp_demo()
