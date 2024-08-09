# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import time as t
import numpy as np
import pytensor
import pytensor.tensor as pt
import pymc
from functools import partial
from itertools import product

import gerabaldi
from gerabaldi.models import *

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt

import stratcona

BOLTZ_EV = 8.617e-5

SHOW_PLOTS = True


def electromigration_model_comparison():
    analyze_prior = True
    run_bed_analysis = False
    simulated_data_mode = 'model'

    mb = stratcona.ModelBuilder(mdl_name='EMMultiSample')
    mb2 = stratcona.ModelBuilder(mdl_name='EMParallelVariability')

    percent_bound, min_lifespan = 99.9, 10 * 8760 # 8760 hours per year on average

    '''
    ===== 2) Model comparison =====
    For our inference model we will use the classic electromigration model based on Black's equation. The current
    density is based on the custom IDFBCAMP chip designed by the Ivanov SoC lab.
    
    Physical parameters that are not directly related to wear-out are assumed to be already characterized
    (e.g., threshold voltage) and so are treated as fixed parameters, only the electromigration specific parameters form
    the latent variable space we infer.
    
    Here we compare two different ways of specifying inherent variability within the model:
    1. The variability of the observable uses an identical model to the mean model, a whole
    separate set of latent variables is used and combined to model the variability
    2. Each latent variable is sampled per observed device, thus variability stems from the mean value estimates and
    the variability of the observation only represents the measurement errors
    '''
    num_devices = 5
    vth_typ, i_base, wire_area = 0.320, 0.8, 1.024 * 1000 * 1000 # Wire area is in nm^2

    # Express wire current density as a function of the number of transistors and the voltage applied
    def j_n(n_fins, vdd):
        return pt.where(pt.gt(vdd, vth_typ), n_fins * i_base * ((vdd - vth_typ) ** 2), 0.0)

    # The classic model for electromigration failure estimates, DOI: 10.1109/T-ED.1969.16754
    def em_blacks_equation(jn_wire, temp, em_n, em_eaa):
        return (wire_area / (jn_wire ** em_n)) * np.exp((em_eaa * 0.01) / (BOLTZ_EV * temp))
    def em_blacks_equation_var(jn_wire, temp, em_n_var, em_eaa_var):
        return (wire_area / (jn_wire ** em_n_var)) * np.exp((em_eaa_var * 0.01) / (BOLTZ_EV * temp))

    # Add some general variability to electromigration sensor line failure times corresponding to 7% of the average time
    def em_variability(em_ttf, em_var):
        return pt.abs(em_var * em_ttf)

    # Now correspond the degradation mechanisms to the output values
    mb.add_dependent_variable('jn_wire', partial(j_n, n_fins=np.array([24])))
    mb.add_dependent_variable('em_ttf', em_blacks_equation)
    mb.add_dependent_variable('em_variability', partial(em_variability, em_var=np.array([0.07])))
    mb.set_variable_observed('em_ttf', variability='em_variability')

    mb.add_latent_variable('em_n', pymc.Normal, {'mu': 1.8, 'sigma': 0.3})
    mb.add_latent_variable('em_eaa', pymc.Normal, {'mu': 2, 'sigma': 0.3})

    mb.define_experiment_params(
        ['vdd', 'temp'], simultaneous_experiments=['t1'],
        samples_per_observation={'em_ttf': num_devices})
    tm = stratcona.TestDesignManager(mb)


    mb2.add_dependent_variable('jn_wire', partial(j_n, n_fins=np.array([24])))
    mb2.add_dependent_variable('em_ttf', em_blacks_equation)
    mb2.add_dependent_variable('em_ttf_var', em_blacks_equation_var)
    mb2.set_variable_observed('em_ttf', variability='em_ttf_var')

    mb2.add_latent_variable('em_n', pymc.Normal, {'mu': 1.8, 'sigma': 0.3})
    mb2.add_latent_variable('em_eaa', pymc.Normal, {'mu': 2, 'sigma': 0.3})
    mb2.add_latent_variable('em_n_var', pymc.TruncatedNormal, {'mu': 0.05, 'sigma': 0.05, 'lower': 0.0, 'upper': 1.0})
    mb2.add_latent_variable('em_eaa_var', pymc.TruncatedNormal, {'mu': 0.05, 'sigma': 0.05, 'lower': 0.0, 'upper': 1.0})

    mb2.define_experiment_params(
        ['vdd', 'temp'], simultaneous_experiments=['t1'],
        samples_per_observation={'em_ttf': num_devices})
    tm2 = stratcona.TestDesignManager(mb2)

    # Can visualize the prior model and see how inference is required to achieve the required predictive confidence.
    if analyze_prior:
        tm.set_experiment_conditions({'t1': {'vdd': 0.85, 'temp': 350}})
        tm.examine('prior_predictive')
        tm.override_func('life_sampler', tm._compiled_funcs['obs_sampler'])
        tm.set_experiment_conditions({'t1': {'vdd': 0.8, 'temp': 330}})
        estimate = tm.estimate_reliability(percent_bound, num_samples=30_000)
        print(f"Estimated {percent_bound}% upper credible lifespan: {estimate} hours")
        tm2.set_experiment_conditions({'t1': {'vdd': 0.85, 'temp': 350}})
        tm2.examine('prior_predictive')
        tm2.override_func('life_sampler', tm2._compiled_funcs['obs_sampler'])
        tm2.set_experiment_conditions({'t1': {'vdd': 0.8, 'temp': 330}})
        estimate = tm2.estimate_reliability(percent_bound, num_samples=30_000)
        print(f"Parallel Var: Estimated {percent_bound}% upper credible lifespan: {estimate} hours")
        if SHOW_PLOTS:
            plt.show()


    temps = [300, 325, 350, 375, 400]
    volts = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    permute_conds = product(temps, volts)
    possible_tests = [{'t1': {'vdd': v, 'temp': t}} for t, v in permute_conds]
    exp_sampler = stratcona.assistants.iterator.iter_sampler(possible_tests)


    if run_bed_analysis:
        tm.curr_rig = 3.2

        # Compile the lifespan function
        field_vdd, field_temp = 0.8, 345
        em_n, em_eaa = pt.dvector('em_n'), pt.dvector('em_eaa')
        fail_time = em_blacks_equation(j_n(24, field_vdd), field_temp, em_n, em_eaa)
        life_func = pytensor.function([em_n, em_eaa], fail_time)
        tm.override_func('life_func', life_func)
        tm.set_upper_credible_target(10 * 8760, 99.9)

        # Run the experimental design analysis
        results = tm.determine_best_test(exp_sampler, num_tests_to_eval=3,
                                         num_obs_samples_per_test=300, num_ltnt_samples_per_test=300)

    selected_test = {'t1': {'vdd': 0.85, 'temp': 350}}
    if simulated_data_mode == 'model':
        # Generate the simulated data points using the same model type as the inference model to allow for validation,
        # since the target posterior is known in this case
        sim_vdd, sim_temp = selected_test['t1']['vdd'], selected_test['t1']['temp']
        em_n, em_eaa = pt.dvector('em_n'), pt.dvector('em_eaa')
        fail_time = em_blacks_equation(j_n(24, sim_vdd), sim_temp, em_n, em_eaa)
        life_func = pytensor.function([em_n, em_eaa], fail_time)

        rng = np.random.default_rng()
        n_sampled = rng.normal(1.65, 0.05, num_devices)
        eaa_sampled = rng.normal(2.05, 0.07, num_devices)
        base_fails = life_func(n_sampled, eaa_sampled)
        fails = rng.normal(base_fails, 0.05 * base_fails)
    else:
        # Use hard coded data
        fails = [120_000, 98_000, 456_000, 400_000, 234_000]

    print(f"Simulated failure times: {fails}")

    '''
    ===== 6) Perform model inference =====
    Update our model based on the emulated test data. First need to extract the failure times from the measurements of
    fail states.
    '''
    tm.set_experiment_conditions({'t1': {'vdd': selected_test['t1']['vdd'], 'temp': selected_test['t1']['temp']}})
    observed = {'t1': {'em_ttf': np.array(fails)}}
    print(tm.get_priors(for_user=True))
    tm.infer_model(observed)
    #tm.infer_model_custom_algo(observed)


if __name__ == '__main__':
    electromigration_model_comparison()