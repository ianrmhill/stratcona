# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from pprint import pprint
import jax
import jax.numpy as jnp
import jax.random as rand
import numpyro
import numpyro.distributions as dists

import gerabaldi
from gerabaldi.models import *

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt

import stratcona

BOLTZ_EV = 8.617e-5
CELSIUS_TO_KELVIN = 273.15

SHOW_PLOTS = True


def tddb_inference():
    numpyro.set_host_device_count(4)

    ### JEDEC HTOL use of Arrhenius ###
    # As seen in JEDEC's JEP-122, acceleration factor = FIT at temp 1 / FIT at temp 2
    # This form gives the field-use failure rate given the accelerated test failure rate
    def differential_arrhenius_equation(fit_exp, e_aa, k, temp_1, temp_2):
        return fit_exp / jnp.exp((-e_aa / k) * ((1 / temp_1) - (1 / temp_2)))

    ### Here's the classic HTOL prediction technique ###
    # With no failures during HTOL, we assume the first failure occurs right after the test ends, thus 1 fail in 1000 hours
    fit_exp = 1 / 1000
    # JEDEC guidance gives Eaa = 0.7 for intrinsic breakdown
    e_aa = 0.7
    # Let's say we test at 125C and our in-field expected temp is 30 above room temp, or 55C
    temp_1, temp_2 = 125 + CELSIUS_TO_KELVIN, 55 + CELSIUS_TO_KELVIN
    # Now we can get the expected FIT at field use conditions
    field_fit = differential_arrhenius_equation(fit_exp, e_aa, BOLTZ_EV, temp_1, temp_2)
    print(f'In-field failure rate: {field_fit / 8760} failures per year, i.e., product will last for {1 / (field_fit * 8760)} years.')

    ### Bayesian use of Arrhenius: DOESN'T MAKE SENSE SINCE FIT IS A STATISTIC NOT AN OBSERVATION ###
    run_arrhenius = False
    if run_arrhenius:
        mb = stratcona.SPMBuilder(mdl_name='Arrhenius')

        # The original Arrhenius equation for reaction rate
        def arrhenius_equation(a, e_aa, k, temp):
            return (a * 1e8) * jnp.exp(-e_aa / (k * temp))

        mb.add_dependent('arrhenius', arrhenius_equation)
        mb.add_measured('fit', dists.Normal, {'loc': 'arrhenius', 'scale': 'fit_var'}, 1)

        var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.01)])
        mb.add_hyperlatent('fit_var_s', dists.Normal, {'loc': 4, 'scale': 0.7}, var_tf)
        mb.add_latent('fit_var', dists.HalfNormal, {'scale': 'fit_var_s'})

        mb.add_hyperlatent('e_aa_l', dists.Normal, {'loc': 0.7, 'scale': 0.02})
        mb.add_hyperlatent('e_aa_s', dists.HalfNormal, {'scale': 0.003})
        mb.add_latent('e_aa', dists.Normal, {'loc': 'e_aa_l', 'scale': 'e_aa_s'})

        mb.add_hyperlatent('a_l', dists.TruncatedNormal, {'loc': 10, 'scale': 2}, fixed_prms={'low': 0})
        mb.add_hyperlatent('a_s', dists.Normal, {'loc': 2, 'scale': 0.5}, dists.transforms.SoftplusTransform())
        mb.add_latent('a', dists.Normal, {'loc': 'e_aa_l', 'scale': 'e_aa_s'})

        mb.add_params(k=BOLTZ_EV)

        mb.add_predictor('field_life', lambda fit: 1 / (fit * 8760), {'temp': 55 + CELSIUS_TO_KELVIN})

        am = stratcona.AnalysisManager(mb.build_model(), rng_seed=92893488)
        test = stratcona.ReliabilityTest({'e0': {'lot': 1, 'chp': 1}}, {'e0': {'temp': 125 + CELSIUS_TO_KELVIN}})
        am.set_test_definition(test)

        # Look at the model under the prior distribution, what is the possible range of FIT values we might see in the test
        # and in the field?
        am.examine_model(['meas', 'hyls'])
        plt.show()

        req = stratcona.ReliabilityRequirement('lbci', 99, 10 * 8760)
        am.relreq = req
        am.evaluate_reliability('field_life')

        # Given an observed FIT, infer the model and update the parameter beliefs
        observed_fit = {'e0': {'fit': 1 / 2900}}
        am.do_inference(observed_fit)
        pprint(am.relmdl.hyl_beliefs)

        # Re-examine the model predictions post inference
        #am.examine_model(['meas', 'hyls'])
        #plt.show()

        # Evaluate reliability
        am.evaluate_reliability('field_life')


    ### Second example case study ###
    run_e_model = False
    if run_e_model:
        mb = stratcona.SPMBuilder(mdl_name='E-Model')

        def e_model_ttf(a_0, e_aa, k, temp):
            ttf_hours = 1e-5 * a_0 * jnp.exp(e_aa / (k * temp))
            ttf_years = ttf_hours / 8760
            return ttf_years

        mb.add_hyperlatent('a_0_nom', dists.Normal, {'loc': 2.2, 'scale': 1}, transform=dists.transforms.SoftplusTransform())
        a_0_var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
        mb.add_hyperlatent('a_0_var', dists.Normal, {'loc': 3, 'scale': 1.5}, transform=a_0_var_tf)
        mb.add_latent('a_0', nom='a_0_nom', dev='a_0_var', chp=None, lot=None)

        mb.add_hyperlatent('e_aa_nom', dists.Normal, {'loc': 0.7, 'scale': 0.03})
        e_aa_var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.001)])
        mb.add_hyperlatent('e_aa_var', dists.Normal, {'loc': 2, 'scale': 1}, transform=e_aa_var_tf)
        mb.add_latent('e_aa', nom='e_aa_nom', dev='e_aa_var')

        mb.add_params(k=BOLTZ_EV, fit_var=2)

        mb.add_dependent('ttf_base', e_model_ttf)
        mb.add_measured('ttf', dists.Normal, {'loc': 'ttf_base', 'scale': 'fit_var'}, 1)

        mb.add_predictor('field_life', lambda ttf: ttf, {'temp': 55 + CELSIUS_TO_KELVIN})

        am = stratcona.AnalysisManager(mb.build_model(), rng_seed=6289383)
        test = stratcona.ReliabilityTest({'e0': {'lot': 1, 'chp': 1, 'ttf': 10}}, {'e0': {'temp': 125 + CELSIUS_TO_KELVIN}})
        am.set_test_definition(test)

        #am.examine_model(['meas', 'hyls'])
        #plt.show()

        req = stratcona.ReliabilityRequirement('lbci', 99, 10)
        am.relreq = req
        am.evaluate_reliability('field_life')

        # Given some observed TTFs, infer the model and update the parameter beliefs
        pprint(am.relmdl.hyl_beliefs)
        observed_fit = {'e0': {'ttf': jnp.array([22.4, 48.3, 72.5, 20.2, 44.2, 34.5, 19.3, 28.4, 76.4, 32.1])}}
        am.do_inference(observed_fit)
        pprint(am.relmdl.hyl_beliefs)

        am.evaluate_reliability('field_life')


    # According to JEP-122, the 0.7V figure is for intrinsic breakdown (TDDB), so this example should handle TDDB

    # Simulate TDDB distributions using all four classic TDDB models, then infer a model using both Bayesian and
    # frequentist techniques to compare
    sim_data = True
    if sim_data:
        mb_e = stratcona.SPMBuilder(mdl_name='E-Model')

        def e_model_ttf(a_o, e_aa, k, temp):
            ttf_hours = 1e-5 * a_o * jnp.exp(e_aa / (k * temp))
            ttf_years = ttf_hours / 8760
            return ttf_years

        mb_e.add_params(a_o_nom=2.8, a_o_dev=0.1, a_o_chp=0.02, a_o_lot=0.1)
        mb_e.add_latent('a_o', nom='a_o_nom', dev='a_o_dev', chp='a_o_chp', lot='a_o_lot')
        mb_e.add_params(e_aa_nom=0.69, e_aa_dev=0.01, e_aa_chp=0.01, e_aa_lot=0.02)
        mb_e.add_latent('e_aa', nom='e_aa_nom', dev='e_aa_dev', chp='e_aa_chp', lot='e_aa_lot')
        mb_e.add_params(k=BOLTZ_EV, fit_var=1)
        mb_e.add_dependent('ttf_base', e_model_ttf)
        mb_e.add_measured('ttf', dists.Normal, {'loc': 'ttf_base', 'scale': 'fit_var'}, 10)

        am_e = stratcona.AnalysisManager(mb_e.build_model(), rng_seed=1299323)
        test = stratcona.ReliabilityTest({'e0': {'lot': 1, 'chp': 1, 'ttf': 10}}, {'e0': {'temp': 125 + CELSIUS_TO_KELVIN}})
        am_e.set_test_definition(test)
        e_fails = am_e.sim_test_measurements()
        print(e_fails)


    # We will compare to "A new clustering-function-based formulation of temporal and spatial
    # clustering model involving area scaling and its application to parameter extraction", an IBM paper from IRPS 2024
    # Multilayer variability modelling of the gate thickness should immediately solve the problems outlined in the paper


    # This example must show the benefits of Bayesian inference in reasoning about physical wear-out models, demo how
    # historical knowledge incorporation, explicit uncertainty reasoning, stochastic variability modelling, and
    # appropriate reasoning with limited data all
    # allow for a more realistic and effective assessment of reliability. Compare to point estimate fitting via
    # regression models, try and follow some frequentist paper.


    # Simulate four groups of two data sets, one with less observed data points, one with lots of points for each
    # of the four commonly used TDDB models

    def tddb_e(a_0, gamma, e_ox, e_aa, k, temp):
        return a_0 * jnp.exp(-gamma * e_ox) * jnp.exp(e_aa / (k * temp))

    def e_ox(v_gs, t_ox):
        return v_gs / t_ox

    def gamma(a, k, temp):
        return a / (k * temp)



if __name__ == '__main__':
    tddb_inference()
