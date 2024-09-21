# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from pprint import pprint
import jax
import jax.numpy as jnp
import jax.random as rand
import numpy as np
import numpyro as npyro
import numpyro.distributions as dists

from functools import partial
from scipy.optimize import curve_fit
from numpy.linalg import cholesky
import reliability
import pandas as pd

import seaborn as sb
from matplotlib import pyplot as plt

import gracefall
import gerabaldi
from gerabaldi.models import *

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stratcona

BOLTZ_EV = 8.617e-5
CELSIUS_TO_KELVIN = 273.15

SHOW_PLOTS = True


def tddb_inference():
    npyro.set_host_device_count(4)

    ### JEDEC HTOL use of Arrhenius ###
    run_htol_est = False
    if run_htol_est:
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

    mb.add_params(k=BOLTZ_EV, fit_var=0.1)

    mb.add_dependent('ttf_base', e_model_ttf)
    mb.add_measured('ttf', dists.Normal, {'loc': 'ttf_base', 'scale': 'fit_var'}, 10)

    mb.add_predictor('field_life', lambda ttf: ttf, {'temp': 55 + CELSIUS_TO_KELVIN})

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=6289383)

    # Data manipulation helper function
    def convert(vals):
        return vals['ttf'].flatten()

    weibull_analysis = True
    if weibull_analysis:
        ## Define the simulation test
        num_devs, num_chps, num_lots = 8, 5, 3
        th = 130 + CELSIUS_TO_KELVIN
        test_130 = stratcona.ReliabilityTest({'e': {'lot': num_lots, 'chp': num_chps}}, {'e': {'temp': th, 'vg': 1.1}})
        # Weibull CDF
        def CDF(x, k, L):
            return 1 - jnp.exp(- (x / L) ** k)

        # Helper function to convert likelihoods to alpha values that result in clear plots
        def likelihood_to_alpha(probs, max_alpha=0.5):
            min_alpha = 0.1
            alphas = min_alpha + ((probs / jnp.max(probs)) * (max_alpha - min_alpha))
            return alphas

        # SPM for Weibull analysis
        mb_w = stratcona.SPMBuilder(mdl_name='weibull-2p')
        mb_w.add_hyperlatent('k_nom', dists.Normal, {'loc': 1.8, 'scale': 0.02}, transform=dists.transforms.SoftplusTransform())
        mb_w.add_hyperlatent('sc_nom', dists.Normal, {'loc': 1.9, 'scale': 0.02}, transform=dists.transforms.SoftplusTransform())
        var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.01)])
        #mb_w.add_hyperlatent('k_dev', dists.Normal, {'loc': 2, 'scale': 0.2}, transform=var_tf)
        #mb_w.add_hyperlatent('sc_dev', dists.Normal, {'loc': 2, 'scale': 0.2}, transform=var_tf)
        mb_w.add_hyperlatent('k_lot', dists.Normal, {'loc': 6, 'scale': 0.2}, transform=var_tf)
        mb_w.add_hyperlatent('sc_lot', dists.Normal, {'loc': 6, 'scale': 0.2}, transform=var_tf)
        #mb_w.add_latent('k', nom='k_nom', dev='k_dev', chp=None, lot='k_lot')
        #mb_w.add_latent('sc', nom='sc_nom', dev='sc_dev', chp=None, lot='sc_lot')
        mb_w.add_latent('k', nom='k_nom', dev=None, chp=None, lot='k_lot')
        mb_w.add_latent('sc', nom='sc_nom', dev=None, chp=None, lot='sc_lot')
        mb_w.add_dependent('k_pos', lambda k: jnp.log(1 + jnp.exp(k)))
        mb_w.add_dependent('sc_pos', lambda sc: jnp.log(1 + jnp.exp(sc)))
        mb_w.add_measured('ttf', dists.Weibull, {'concentration': 'k_pos', 'scale': 'sc_pos'}, num_devs)

        am_w = stratcona.AnalysisManager(mb_w.build_model(), rng_seed=92733429)

        am_w.set_test_definition(test_130)
        # TODO: Return the three values for k_pos and sc_pos used to generate each series, then can plot the 'true' functions
        ttfs = am_w.sim_test_measurements()
        ttfs_0 = {'e': {'ttf': ttfs['e']['ttf'][:, :, :, 0]}}
        ttfs_1 = {'e': {'ttf': ttfs['e']['ttf'][:, :, :, 1]}}
        ttfs_2 = {'e': {'ttf': ttfs['e']['ttf'][:, :, :, 2]}}

        test_130_sing = stratcona.ReliabilityTest({'e': {'lot': 1, 'chp': 1, 'ttf': 1}}, {'e': {'temp': th, 'vg': 1.1}})
        k1, k2, k3, k4 = rand.split(rand.key(428027234), 4)
        eval_sites = ['e_ttf_k_dev', 'e_ttf_sc_dev', 'e_k_lot', 'e_sc_lot',
                      'k_nom', 'sc_nom', 'k_dev', 'sc_dev', 'k_lot', 'sc_lot']

        am_w.relmdl.hyl_beliefs = {'k_nom': {'loc': 2.0, 'scale': 1.5}, 'k_dev': {'loc': 5, 'scale': 3}, 'k_lot': {'loc': 12, 'scale': 3},
                                   'sc_nom': {'loc': 1.8, 'scale': 0.8}, 'sc_dev': {'loc': 5, 'scale': 3}, 'sc_lot': {'loc': 13, 'scale': 5}}
        prm_samples = am_w.relmdl.sample(k1, test_130_sing, 400)
        ltnt_vals = {site: data for site, data in prm_samples.items() if site in eval_sites}
        pri_probs = jnp.exp(am_w.relmdl.logp(k2, test_130_sing, ltnt_vals, prm_samples))
        pri_probs = likelihood_to_alpha(pri_probs, 0.4).flatten()

        x = jnp.logspace(-2, 1, 50)
        pri_fits = CDF(x, prm_samples['e_ttf_k_pos'], prm_samples['e_ttf_sc_pos'])

        am_w.do_inference(ttfs, test_130)
        print(am_w.relmdl.hyl_beliefs)

        prm_samples = am_w.relmdl.sample(k3, test_130_sing, 400)
        ltnt_vals = {site: data for site, data in prm_samples.items() if site in eval_sites}
        pst_probs = jnp.exp(am_w.relmdl.logp(k4, test_130_sing, ltnt_vals, prm_samples))
        pst_probs = likelihood_to_alpha(pst_probs, 0.4).flatten()

        pst_fits = CDF(x, prm_samples['e_ttf_k_pos'], prm_samples['e_ttf_sc_pos'])

        # Frequentist analysis of the failure data
        fails_0, fails_1, fails_2 = convert(ttfs_0['e']), convert(ttfs_1['e']), convert(ttfs_2['e'])
        fails = jnp.concatenate((fails_0, fails_1, fails_2))
        lots = jnp.array([0, 1, 2]).repeat(num_devs * num_chps)

        mean, var = fails.mean(), fails.var()
        fit_full = reliability.Fitters.Fit_Weibull_2P(fails.tolist(), show_probability_plot=False)
        fit_0 = reliability.Fitters.Fit_Weibull_2P(fails_0.tolist(), show_probability_plot=False)
        fit_1 = reliability.Fitters.Fit_Weibull_2P(fails_1.tolist(), show_probability_plot=False)
        fit_2 = reliability.Fitters.Fit_Weibull_2P(fails_2.tolist(), show_probability_plot=False)


        # Get data for the weibull fit plot
        x = jnp.logspace(-2, 1, 50)
        # NOTE: Some of these fits are bad! This is good for the Bayesian approach!
        fit_fails = CDF(x, fit_full.beta, fit_full.alpha)
        fit_fails_0 = CDF(x, fit_0.beta, fit_0.alpha)
        fit_fails_1 = CDF(x, fit_1.beta, fit_1.alpha)
        fit_fails_2 = CDF(x, fit_2.beta, fit_2.alpha)

        n = len(fails)
        i = jnp.arange(1, n + 1)
        fail_order = (i - 0.5) / (n + 0.25)

        n = len(fails_0)
        i = jnp.arange(1, n + 1)
        fail_order_s = (i - 0.5) / (n + 0.25)

        # Generate a Weibull plot!
        srtd_inds = jnp.argsort(fails)
        fails = fails[srtd_inds]
        srtd_lots = lots[srtd_inds]

        fails_0 = np.sort(fails_0)
        fails_1 = np.sort(fails_1)
        fails_2 = np.sort(fails_2)

        n0 = np.argwhere(srtd_lots != 0)
        f0 = jnp.delete(fails, n0)
        x0 = jnp.delete(fail_order, n0)
        n1 = np.argwhere(srtd_lots != 1)
        f1 = jnp.delete(fails, n1)
        x1 = jnp.delete(fail_order, n1)
        n2 = np.argwhere(srtd_lots != 2)
        f2 = jnp.delete(fails, n2)
        x2 = jnp.delete(fail_order, n2)

        # Functions to correctly set up the axis scales
        ax_fwdy = lambda p: jnp.log(jnp.fmax(1e-20, -jnp.log(jnp.fmax(1e-20, 1 - p))))
        ax_bcky = lambda q: 1 - jnp.exp(-jnp.exp(q))
        ax_fwdx = lambda x: jnp.log(jnp.fmax(1e-20, x))
        ax_bckx = lambda y: jnp.exp(y)

        sb.set_context('notebook')
        fig, p = plt.subplots(1, 1)
        p.grid()
        # Plot the probabilistic fits
        for i in range(len(pri_fits)):
            p.plot(x, pri_fits[i].flatten(), alpha=float(pri_probs[i]), color='yellowgreen')
        for i in range(len(pst_fits)):
            p.plot(x, pst_fits[i].flatten(), alpha=float(pst_probs[i]), color='lightseagreen')

        p.plot(x, fit_fails, color='indigo', linewidth=2)
        p.plot(x, fit_fails_0, color='orchid', linestyle='--', linewidth=2)
        p.plot(x, fit_fails_1, color='darkorchid', linestyle='--', linewidth=2)
        p.plot(x, fit_fails_2, color='mediumvioletred', linestyle='--', linewidth=2)

        p.plot(fails_0, fail_order_s, color='orchid', linestyle='', marker='.', markersize=8)
        p.plot(fails_1, fail_order_s, color='darkorchid', linestyle='', marker='.', markersize=8)
        p.plot(fails_2, fail_order_s, color='mediumvioletred', linestyle='', marker='.', markersize=8)

        p.plot(f0, x0, color='orchid', linestyle='', marker='.', markersize=6)
        p.plot(f1, x1, color='darkorchid', linestyle='', marker='.', markersize=6)
        p.plot(f2, x2, color='mediumvioletred', linestyle='', marker='.', markersize=6)
        #p.plot(fails, fail_order, color='black', linestyle='', marker='.', markersize=5)

        p.set_xscale('function', functions=(ax_fwdx, ax_bckx))
        ln_min, ln_max = jnp.log(min(fails)), jnp.log(max(fails))
        lim_l = jnp.exp(ln_min - (0.01 * (ln_max - ln_min)))
        lim_h = jnp.exp(ln_max + (0.05 * (ln_max - ln_min)))
        p.set_xlim(lim_l, lim_h)
        p.set_yscale('function', functions=(ax_fwdy, ax_bcky))
        p.set_ylim(0.01, 0.99)
        weibull_ticks = [0.01, 0.02, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.96, 0.99]
        p.set_yticks(weibull_ticks)

        p.set_xlabel('Time to Failure (years)')
        p.set_ylabel('CDF [ln(ln(1-F))]')
        plt.show()

    run_e_model = False
    if run_e_model:
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

    # Simulate TDDB distributions using all four classic TDDB models, then infer a model using both Bayesian and
    # frequentist techniques to compare. Only temperature acceleration considered.
    sim_data = False
    if sim_data:
        # Define the simulation test
        tl, tm, th = 85 + CELSIUS_TO_KELVIN, 125 + CELSIUS_TO_KELVIN, 145 + CELSIUS_TO_KELVIN
        test_s = stratcona.ReliabilityTest(
            {'el': {'lot': 1, 'chp': 1}, 'em': {'lot': 1, 'chp': 1}, 'eh': {'lot': 1, 'chp': 1}},
            {'el': {'temp': tl, 'vg': 1.1}, 'em': {'temp': tm, 'vg': 1.1}, 'eh': {'temp': th, 'vg': 1.1}})
        test_l = stratcona.ReliabilityTest(
            {'el': {'lot': 5, 'chp': 5}, 'em': {'lot': 5, 'chp': 5}, 'eh': {'lot': 5, 'chp': 5}},
            {'el': {'temp': tl, 'vg': 1.1}, 'em': {'temp': tm, 'vg': 1.1}, 'eh': {'temp': th, 'vg': 1.1}})
        fails_s = {'el': {}, 'em': {}, 'eh': {}}
        ttfs = {'s': {}, 'l': {}}
        fails_l = {'el': {}, 'em': {}, 'eh': {}}
        means, vars = {}, {}

        # E model
        mb_e = stratcona.SPMBuilder(mdl_name='E-Model')

        def e_model_ttf(temp, a_o, e_aa, k):
            ttf_hours = 1e-5 * a_o * jnp.exp(e_aa / (k * temp))
            ttf_years = ttf_hours / 8760
            return ttf_years

        mb_e.add_params(a_o_nom=3.8, a_o_dev=0.05, a_o_chp=0.02, a_o_lot=0.05)
        mb_e.add_latent('a_o', nom='a_o_nom', dev='a_o_dev', chp='a_o_chp', lot='a_o_lot')
        mb_e.add_params(e_aa_nom=0.70, e_aa_dev=0.02, e_aa_chp=0.01, e_aa_lot=0.01)
        mb_e.add_latent('e_aa', nom='e_aa_nom', dev='e_aa_dev', chp='e_aa_chp', lot='e_aa_lot')

        mb_e.add_params(k=BOLTZ_EV, ttf_var=0.001)
        mb_e.add_dependent('ttf_base', e_model_ttf)
        mb_e.add_measured('ttf', dists.Normal, {'loc': 'ttf_base', 'scale': 'ttf_var'}, 10)

        means['e'], vars['e'] = {}, {}
        am_e = stratcona.AnalysisManager(mb_e.build_model(), rng_seed=1295323)
        am_e.set_test_definition(test_s)
        ttfs['s']['e'] = am_e.sim_test_measurements()
        means['e']['el_s'], means['e']['em_s'], means['e']['eh_s'] = ttfs['s']['e']['el']['ttf'].mean(), ttfs['s']['e']['em']['ttf'].mean(), ttfs['s']['e']['eh']['ttf'].mean()
        vars['e']['el_s'], vars['e']['em_s'], vars['e']['eh_s'] = ttfs['s']['e']['el']['ttf'].var(), ttfs['s']['e']['em']['ttf'].var(), ttfs['s']['e']['eh']['ttf'].var()
        fails_s['el']['e'], fails_s['em']['e'], fails_s['eh']['e'] = convert(ttfs['s']['e']['el']), convert(ttfs['s']['e']['em']), convert(ttfs['s']['e']['eh'])
        am_e.set_test_definition(test_l)
        ttfs['l']['e'] = am_e.sim_test_measurements()
        means['e']['el_l'], means['e']['em_l'], means['e']['eh_l'] = ttfs['l']['e']['el']['ttf'].mean(), ttfs['l']['e']['em']['ttf'].mean(), ttfs['l']['e']['eh']['ttf'].mean()
        vars['e']['el_l'], vars['e']['em_l'], vars['e']['eh_l'] = ttfs['l']['e']['el']['ttf'].var(), ttfs['l']['e']['em']['ttf'].var(), ttfs['l']['e']['eh']['ttf'].var()
        fails_l['el']['e'], fails_l['em']['e'], fails_l['eh']['e'] = convert(ttfs['l']['e']['el']), convert(ttfs['l']['e']['em']), convert(ttfs['l']['e']['eh'])

        # 1/E model, reference paper: Temperature Acceleration of Time-Dependent Dielectric Breakdown: 10.1109/16.43668
        mb_inv_e = stratcona.SPMBuilder(mdl_name='Inverse-E-Model')

        def inv_e_model_ttf(temp, t_o, e_b, e_ox, g_o, delta, k):
            tau_o = t_o * jnp.exp((-e_b / k) * ((1 / temp) - (1 / 300)))
            g = g_o * (1 + ((delta / k) * ((1 / temp) - (1 / 300))))
            ttf_hours = 1e6 * tau_o * jnp.exp(g / e_ox)
            ttf_years = ttf_hours / 8760
            return ttf_years

        mb_inv_e.add_params(t_o_nom=6.3, t_o_dev=0.1, t_o_chp=0.05, t_o_lot=0.03)
        mb_inv_e.add_latent('t_o', nom='t_o_nom', dev='t_o_dev', chp='t_o_chp', lot='t_o_lot')
        mb_inv_e.add_params(g_o_nom=3.5, g_o_dev=0.02, g_o_chp=0.01, g_o_lot=0.01)
        mb_inv_e.add_latent('g_o', nom='g_o_nom', dev='g_o_dev', chp='g_o_chp', lot='g_o_lot')
        mb_inv_e.add_params(e_b_nom=3.7, e_b_dev=0.07, e_b_chp=0.01, e_b_lot=0.01)
        mb_inv_e.add_latent('e_b', nom='e_b_nom', dev='e_b_dev', chp='e_b_chp', lot='e_b_lot')
        mb_inv_e.add_params(delta_nom=3.5, delta_dev=0.02, delta_chp=0.01, delta_lot=0.01)
        mb_inv_e.add_latent('delta', nom='delta_nom', dev='delta_dev', chp='delta_chp', lot='delta_lot')
        mb_inv_e.add_params(e_ox_nom=3.2, e_ox_dev=0.02, e_ox_chp=0.01, e_ox_lot=0.01)
        mb_inv_e.add_latent('e_ox', nom='e_ox_nom', dev='e_ox_dev', chp='e_ox_chp', lot='e_ox_lot')

        mb_inv_e.add_params(k=BOLTZ_EV, ttf_var=0.001)
        mb_inv_e.add_dependent('inv_e_ttf', inv_e_model_ttf)
        mb_inv_e.add_measured('ttf', dists.Normal, {'loc': 'inv_e_ttf', 'scale': 'ttf_var'}, 10)

        means['ie'], vars['ie'] = {}, {}
        am_inv_e = stratcona.AnalysisManager(mb_inv_e.build_model(), rng_seed=3229823)
        am_inv_e.set_test_definition(test_s)
        ttfs['s']['ie'] = am_inv_e.sim_test_measurements()
        means['ie']['el_s'], means['ie']['em_s'], means['ie']['eh_s'] = ttfs['s']['ie']['el']['ttf'].mean(), ttfs['s']['ie']['em']['ttf'].mean(), ttfs['s']['ie']['eh']['ttf'].mean()
        vars['ie']['el_s'], vars['ie']['em_s'], vars['ie']['eh_s'] = ttfs['s']['ie']['el']['ttf'].var(), ttfs['s']['ie']['em']['ttf'].var(), ttfs['s']['ie']['eh']['ttf'].var()
        fails_s['el']['ie'], fails_s['em']['ie'], fails_s['eh']['ie'] = convert(ttfs['s']['ie']['el']), convert(ttfs['s']['ie']['em']), convert(ttfs['s']['ie']['eh'])
        am_inv_e.set_test_definition(test_l)
        ttfs['l']['ie'] = am_inv_e.sim_test_measurements()
        means['ie']['el_l'], means['ie']['em_l'], means['ie']['eh_l'] = ttfs['l']['ie']['el']['ttf'].mean(), ttfs['l']['ie']['em']['ttf'].mean(), ttfs['l']['ie']['eh']['ttf'].mean()
        vars['ie']['el_l'], vars['ie']['em_l'], vars['ie']['eh_l'] = ttfs['l']['ie']['el']['ttf'].var(), ttfs['l']['ie']['em']['ttf'].var(), ttfs['l']['ie']['eh']['ttf'].var()
        fails_l['el']['ie'], fails_l['em']['ie'], fails_l['eh']['ie'] = convert(ttfs['l']['ie']['el']), convert(ttfs['l']['ie']['em']), convert(ttfs['l']['ie']['eh'])

        # V model
        # Note: With constant voltage, V model ends up having identical form to the E model just with a prescale, so
        #       it is not used here
        #mb_v = stratcona.SPMBuilder(mdl_name='V-Model')

        #def v_model_ttf(temp, a_o, beta, e_aa, vg, k):
        #    ttf_hours = 1e-5 * a_o * jnp.exp(-beta * vg) * jnp.exp(e_aa / (k * temp))
        #    ttf_years = ttf_hours / 8760
        #    return ttf_years

        #mb_v.add_params(a_o_nom=9.8, a_o_dev=0.04, a_o_chp=0.02, a_o_lot=0.04)
        #mb_v.add_latent('a_o', nom='a_o_nom', dev='a_o_dev', chp='a_o_chp', lot='a_o_lot')
        #mb_v.add_params(beta_nom=1.1, beta_dev=0.05, beta_chp=0.1, beta_lot=0.05)
        #mb_v.add_latent('beta', nom='beta_nom', dev='beta_dev', chp='beta_chp', lot='beta_lot')
        #mb_v.add_params(e_aa_nom=0.72, e_aa_dev=0.01, e_aa_chp=0.01, e_aa_lot=0.02)
        #mb_v.add_latent('e_aa', nom='e_aa_nom', dev='e_aa_dev', chp='e_aa_chp', lot='e_aa_lot')

        #mb_v.add_params(k=BOLTZ_EV, ttf_var=0.001)
        #mb_v.add_dependent('ttf_base', v_model_ttf)
        #mb_v.add_measured('ttf', dists.Normal, {'loc': 'ttf_base', 'scale': 'ttf_var'}, 10)

        #means['v'], vars['v'] = {}, {}
        #am_v = stratcona.AnalysisManager(mb_v.build_model(), rng_seed=1299323)
        #am_v.set_test_definition(test_s)
        #ttfs['s']['v'] = am_v.sim_test_measurements()
        #means['v']['el_s'], means['v']['em_s'], means['v']['eh_s'] = ttfs['s']['v']['el']['ttf'].mean(), ttfs['s']['v']['em']['ttf'].mean(), ttfs['s']['v']['eh']['ttf'].mean()
        #vars['v']['el_s'], vars['v']['em_s'], vars['v']['eh_s'] = ttfs['s']['v']['el']['ttf'].var(), ttfs['s']['v']['em']['ttf'].var(), ttfs['s']['v']['eh']['ttf'].var()
        #fails_s['el']['v'], fails_s['em']['v'], fails_s['eh']['v'] = convert(ttfs['s']['v']['el']), convert(ttfs['s']['v']['em']), convert(ttfs['s']['v']['eh'])
        #am_v.set_test_definition(test_l)
        #ttfs['l']['v'] = am_v.sim_test_measurements()
        #means['v']['el_l'], means['v']['em_l'], means['v']['eh_l'] = ttfs['l']['v']['el']['ttf'].mean(), ttfs['l']['v']['em']['ttf'].mean(), ttfs['l']['v']['eh']['ttf'].mean()
        #vars['v']['el_l'], vars['v']['em_l'], vars['v']['eh_l'] = ttfs['l']['v']['el']['ttf'].var(), ttfs['l']['v']['em']['ttf'].var(), ttfs['l']['v']['eh']['ttf'].var()
        #fails_l['el']['v'], fails_l['em']['v'], fails_l['eh']['v'] = convert(ttfs['l']['v']['el']), convert(ttfs['l']['v']['em']), convert(ttfs['l']['v']['eh'])

        # Power law model, reference: Interplay of voltage and temperature acceleration of oxide breakdown for
        # ultra-thin gate oxides: https://doi.org/10.1016/S0038-1101(02)00151-X
        mb_p = stratcona.SPMBuilder(mdl_name='Power-Law-Model')

        def p_model_ttf(temp, t_o, n, a, b, vg):
            ttf_hours = 1e4 * t_o * (vg ** -n) * jnp.exp(((a * 100) / temp) + ((b * 10000) / (temp ** 2)))
            ttf_years = ttf_hours / 8760
            return ttf_years

        mb_p.add_params(t_o_nom=5.4, t_o_dev=0.1, t_o_chp=0.03, t_o_lot=0.05)
        mb_p.add_latent('t_o', nom='t_o_nom', dev='t_o_dev', chp='t_o_chp', lot='t_o_lot')
        mb_p.add_params(n_nom=2.4, n_dev=0.01, n_chp=0.01, n_lot=0.01)
        mb_p.add_latent('n', nom='n_nom', dev='n_dev', chp='n_chp', lot='n_lot')
        mb_p.add_params(a_nom=2.3, a_dev=0.01, a_chp=0.01, a_lot=0.01)
        mb_p.add_latent('a', nom='a_nom', dev='a_dev', chp='a_chp', lot='a_lot')
        mb_p.add_params(b_nom=6.4, b_dev=0.01, b_chp=0.01, b_lot=0.01)
        mb_p.add_latent('b', nom='b_nom', dev='b_dev', chp='b_chp', lot='b_lot')

        mb_p.add_params(k=BOLTZ_EV, ttf_var=0.001)
        mb_p.add_dependent('ttf_base', p_model_ttf)
        mb_p.add_measured('ttf', dists.Normal, {'loc': 'ttf_base', 'scale': 'ttf_var'}, 10)

        means['p'], vars['p'] = {}, {}
        am_p = stratcona.AnalysisManager(mb_p.build_model(), rng_seed=1299323)
        am_p.set_test_definition(test_s)
        ttfs['s']['p'] = am_p.sim_test_measurements()
        means['p']['el_s'], means['p']['em_s'], means['p']['eh_s'] = ttfs['s']['p']['el']['ttf'].mean(), ttfs['s']['p']['em']['ttf'].mean(), ttfs['s']['p']['eh']['ttf'].mean()
        vars['p']['el_s'], vars['p']['em_s'], vars['p']['eh_s'] = ttfs['s']['p']['el']['ttf'].var(), ttfs['s']['p']['em']['ttf'].var(), ttfs['s']['p']['eh']['ttf'].var()
        fails_s['el']['p'], fails_s['em']['p'], fails_s['eh']['p'] = convert(ttfs['s']['p']['el']), convert(ttfs['s']['p']['em']), convert(ttfs['s']['p']['eh'])
        am_p.set_test_definition(test_l)
        ttfs['l']['p'] = am_p.sim_test_measurements()
        means['p']['el_l'], means['p']['em_l'], means['p']['eh_l'] = ttfs['l']['p']['el']['ttf'].mean(), ttfs['l']['p']['em']['ttf'].mean(), ttfs['l']['p']['eh']['ttf'].mean()
        vars['p']['el_l'], vars['p']['em_l'], vars['p']['eh_l'] = ttfs['l']['p']['el']['ttf'].var(), ttfs['l']['p']['em']['ttf'].var(), ttfs['l']['p']['eh']['ttf'].var()
        fails_l['el']['p'], fails_l['em']['p'], fails_l['eh']['p'] = convert(ttfs['l']['p']['el']), convert(ttfs['l']['p']['em']), convert(ttfs['l']['p']['eh'])

        plot_sim_data = True
        if plot_sim_data:

            for k in means:
                print(f"Means of {k}: low s: {means[k]['el_s']}, med s: {means[k]['em_s']}, hi s: {means[k]['eh_s']}, low l: {means[k]['el_l']}, med l: {means[k]['em_l']}, hi l: {means[k]['eh_l']}")
                print(f"Variances of {k}: low s: {vars[k]['el_s']}, med s: {vars[k]['em_s']}, hi s: {vars[k]['eh_s']}, low l: {vars[k]['el_l']}, med l: {vars[k]['em_l']}, hi l: {vars[k]['eh_l']}")

            sb.set_theme(style='ticks', font='Times New Roman')
            sb.set_context('notebook')
            model_colours = ['mediumblue', 'darkviolet', 'lightseagreen', 'deeppink']

            fig, p = plt.subplots(1, 3)
            sb.stripplot(data=fails_s['el'], ax=p[0], palette=model_colours, orient='h')
            p[0].set_title('T=85C')
            sb.stripplot(data=fails_s['em'], ax=p[1], palette=model_colours, orient='h')
            p[1].set_title('T=125C')
            sb.stripplot(data=fails_s['eh'], ax=p[2], palette=model_colours, orient='h')
            p[2].set_title('T=145C')
            for i in range(len(p)):
                p[i].set_yticks([0, 1, 2, 3], labels=['E Model', '1/E Model', 'V Model', 'Power Law Model'])
                p[i].set_ylabel(None)
                p[i].set_xlabel('Time to Failure (years)')
                p[i].set_xscale('log')
            fig.subplots_adjust(hspace=10)

            fig, p = plt.subplots(1, 3)
            sb.stripplot(data=fails_l['el'], ax=p[0], palette=model_colours, orient='h')
            p[0].set_title('T=85C')
            sb.stripplot(data=fails_l['em'], ax=p[1], palette=model_colours, orient='h')
            p[1].set_title('T=125C')
            sb.stripplot(data=fails_l['eh'], ax=p[2], palette=model_colours, orient='h')
            p[2].set_title('T=145C')
            for i in range(len(p)):
                p[i].set_yticks([0, 1, 2], labels=['E Model', '1/E Model', 'Power Law Model'])
                p[i].set_ylabel(None)
                p[i].set_xlabel('Time to Failure (years)')
                p[i].set_xscale('log')
            fig.subplots_adjust(hspace=10)
            plt.show()


        infer_spm = True
        if infer_spm:
            test = stratcona.ReliabilityTest(
                {'e': {'lot': 1, 'chp': 1}},
                {'e': {'temp': 300, 'vg': 1.1}, 'em': {'temp': tm, 'vg': 1.1}, 'eh': {'temp': th, 'vg': 1.1}})
            rng = rand.key(48408)
            k1, k2 = rand.split(rng)
            prm_samples = am.relmdl.sample(k1, test, 100, keep_sites=['a_0_nom', 'e_aa_nom'])
            pri_sample_probs = jnp.exp(am.relmdl.hyl_logp(k2, test, prm_samples))
            pri_sample_probs = pri_sample_probs / (jnp.max(pri_sample_probs) * 2)
            spm_temps = jnp.full((100, 13,), jnp.linspace(300, 420, 13)).T
            pri_spm_vals = e_model_ttf(spm_temps, prm_samples['a_0_nom'], prm_samples['e_aa_nom'], BOLTZ_EV)
            pri_spm_vals = pri_spm_vals.T

            am.set_test_definition(test_s)
            am.do_inference(ttfs['s']['e'])
            pprint(am.relmdl.hyl_beliefs)



            a_o_infd = am.relmdl.hyl_beliefs['a_0_nom']['loc']
            e_aa_infd = am.relmdl.hyl_beliefs['e_aa_nom']['loc']
            ttf_l = e_model_ttf(temp=tl, a_o=a_o_infd, e_aa=e_aa_infd, k=BOLTZ_EV)
            ttf_m = e_model_ttf(temp=tm, a_o=a_o_infd, e_aa=e_aa_infd, k=BOLTZ_EV)
            ttf_h = e_model_ttf(temp=th, a_o=a_o_infd, e_aa=e_aa_infd, k=BOLTZ_EV)

            mean_temps = jnp.linspace(300, 420, 13)
            mean_infd = e_model_ttf(temp=mean_temps, a_o=a_o_infd, e_aa=e_aa_infd, k=BOLTZ_EV)
            print(f'Inference model predicted MTTF: 85C={ttf_l}, 125C={ttf_m}, 145C={ttf_h}')


            rng = rand.key(23498)
            prm_samples = am.relmdl.sample(rng, test, 100, keep_sites=['a_0_nom', 'e_aa_nom'])
            sample_probs = jnp.exp(am.relmdl.hyl_logp(k2, test, prm_samples))
            sample_probs = sample_probs / (jnp.max(sample_probs) * 2)
            spm_vals = e_model_ttf(spm_temps, prm_samples['a_0_nom'], prm_samples['e_aa_nom'], BOLTZ_EV)
            spm_vals, spm_temps = spm_vals.T, spm_temps.T


        fit_models = True
        if fit_models:
            # NOTE: Because the shape of e^x is invariant to the magnitude of x, there are an infinite number of fits
            #       for the data that are equally good as a_o and e_aa can simply slide relative to each other without
            #       changing the model predictions
            temps_s = jnp.repeat(jnp.array([tl, tm, th]), 10)
            temps_l = jnp.repeat(jnp.array([tl, tm, th]), 10 * 5 * 5)

            funcs = {}
            funcs['e'] = partial(e_model_ttf, k=BOLTZ_EV)
            funcs['ie'] = partial(inv_e_model_ttf, k=BOLTZ_EV)
            funcs['p'] = partial(p_model_ttf, vg=1.1)

            fits = {'s': {}, 'l': {}}
            for mdl in ['e', 'ie', 'p']:
                fits['s'][mdl], fits['l'][mdl] = {}, {}
                for sim in ['e', 'ie', 'p']:
                    s_fails = jnp.concatenate((convert(ttfs['s'][sim]['el']), convert(ttfs['s'][sim]['em']), convert(ttfs['s'][sim]['eh'])))
                    l_fails = jnp.concatenate((convert(ttfs['l'][sim]['el']), convert(ttfs['l'][sim]['em']), convert(ttfs['l'][sim]['eh'])))
                    try:
                        fits['s'][mdl][sim] = curve_fit(funcs[mdl], xdata=temps_s, ydata=s_fails)
                        fits['l'][mdl][sim] = curve_fit(funcs[mdl], xdata=temps_l, ydata=l_fails)
                    except RuntimeError:
                        fits['s'][mdl][sim] = None
                        fits['l'][mdl][sim] = None

            #print(fits)

            y_ttfs = {'s': {}, 'l': {}}
            x_temps = jnp.linspace(300, 420, 13)
            y_ttfs['s']['e'] = funcs['e'](x_temps, *fits['s']['e']['e'][0])
            y_ttfs['l']['e'] = funcs['e'](x_temps, *fits['l']['e']['e'][0])
            fit_prms = fits['s']['e']['e']
            e_sampler = dists.MultivariateStudentT(8, fit_prms[0], cholesky(fit_prms[1]))
            rng_key = rand.key(23492)
            print(e_sampler.sample(rng_key, (10,)))

            #y_ttfs['s']['ie'] = funcs['ie'](x_temps, *fits['s']['ie']['e'][0])
            #y_ttfs['l']['ie'] = funcs['ie'](x_temps, *fits['l']['ie']['e'][0])
            y_ttfs['s']['p'] = funcs['p'](x_temps, *fits['s']['p']['e'][0])
            y_ttfs['l']['p'] = funcs['p'](x_temps, *fits['l']['p']['e'][0])

            sb.set_context('notebook')
            fig, p = plt.subplots(1, 1)
            p.grid()
            # If the Bayesian model was inferred plot that first since it adds many series to the plot
            if infer_spm:
                for i in range(100):
                    p.plot(spm_temps[i], pri_spm_vals[i], color='burlywood', linestyle='-', alpha=float(pri_sample_probs[i]))
                    p.plot(spm_temps[i], spm_vals[i], color='royalblue', linestyle='-', alpha=float(sample_probs[i]))

            p.plot(mean_temps, mean_infd, color='crimson')
            s_fails = jnp.concatenate((convert(ttfs['s']['e']['el']), convert(ttfs['s']['e']['em']), convert(ttfs['s']['e']['eh'])))
            p.plot(temps_s, s_fails, color='sienna', linestyle='', marker='.', markersize=15)
            l_fails = jnp.concatenate((convert(ttfs['l']['e']['el']), convert(ttfs['l']['e']['em']), convert(ttfs['l']['e']['eh'])))
            p.plot(temps_l, l_fails, color='black', alpha=0.3, linestyle='', marker='.', markersize=4)

            # Plot the curve used to generate the observed data
            p.plot(x_temps, funcs['e'](x_temps, 3.8, 0.7), color='grey')
            mdl_clrs = ['darkviolet', 'lightseagreen', 'deeppink']
            for i, k in enumerate(y_ttfs['s']):
                p.plot(x_temps, y_ttfs['s'][k], color=mdl_clrs[i], linestyle=':', linewidth=2)
                p.plot(x_temps, y_ttfs['l'][k], color=mdl_clrs[i], linestyle='--', linewidth=2)

            p.set_title('E Model Predicted')
            p.set_xlabel('Temperature (K)')
            p.set_ylabel('Time to Failure (years)')
            p.set_yscale('log')
            plt.show()

            #ttf_l = e_func(temp=tl, a_o=fit[0][0], e_aa=fit[0][1])
            #ttf_m = e_func(temp=tm, a_o=fit[0][0], e_aa=fit[0][1])
            #ttf_h = e_func(temp=th, a_o=fit[0][0], e_aa=fit[0][1])
            #print(f'Fitted model predicted MTTF: 85C={ttf_l}, 125C={ttf_m}, 145C={ttf_h}')




    # We will compare to "A new clustering-function-based formulation of temporal and spatial
    # clustering model involving area scaling and its application to parameter extraction", an IBM paper from IRPS 2024
    # Multilayer variability modelling of the gate thickness should immediately solve the problems outlined in the paper


    # This example must show the benefits of Bayesian inference in reasoning about physical wear-out models, demo how
    # historical knowledge incorporation, explicit uncertainty reasoning, stochastic variability modelling, and
    # appropriate reasoning with limited data all
    # allow for a more realistic and effective assessment of reliability. Compare to point estimate + covariance fitting
    # via regression models, try and follow some frequentist paper.

    def avg_same_time_measurements(measd, average=True, normalize=False):
        avgd = pd.DataFrame()
        t_prev = -10000
        # Average the measurements of the same parameter taken near the same time relative to the total test length
        for prm in measd['param'].unique():
            prm_meas = measd.loc[measd['param'] == prm]
            # Averaging for repeated measurement of devices during each measurement time step
            for lot in range(prm_meas['lot #'].min(), prm_meas['lot #'].max() + 1):
                for chp in range(prm_meas['chip #'].min(), prm_meas['chip #'].max() + 1):
                    for dev in range(prm_meas['device #'].min(), prm_meas['device #'].max() + 1):
                        # For every individual device measured, we find measured times that are close together (+-6 mins)
                        dev_meas = prm_meas.loc[(prm_meas['device #'] == dev) &
                                                (prm_meas['chip #'] == chp) &
                                                (prm_meas['lot #'] == lot)]
                        for t in dev_meas['time']:
                            # Only process each time 'group' once
                            if t <= t_prev - 360 or t >= t_prev + 360:
                                meas_group = dev_meas.loc[(dev_meas['time'] >= t - 360) & (dev_meas['time'] <= t + 360)]
                                # Find the average of all the measurements taken near the specified time
                                avg = meas_group['measured'].mean()
                                dev_meas.loc[dev_meas['time'] == t, 'measured'] = avg
                                # Assemble the new dataframe by appending each averaged row in turn
                                avgd = pd.concat((avgd, dev_meas.loc[dev_meas['time'] == t]), ignore_index=True)
                                t_prev = t
                        # print(f"Finished a device for param {prm}!")
        return avgd

    def diff_from_max_val(measd, return_maxes=False):
        vals = measd.copy()
        for prm in vals['param'].unique():
            prm_data = vals.loc[vals['param'] == prm]
            for lot in range(prm_data['lot #'].min(), prm_data['lot #'].max() + 1):
                for chp in range(prm_data['chip #'].min(), prm_data['chip #'].max() + 1):
                    for dev in range(prm_data['device #'].min(), prm_data['device #'].max() + 1):
                        max_val = prm_data.loc[(prm_data['device #'] == dev) & (prm_data['chip #'] == chp) & (
                                    prm_data['lot #'] == lot), 'measured'].max()
                        if return_maxes:
                            vals.loc[(vals['param'] == prm) & (vals['device #'] == dev) & (vals['chip #'] == chp) & (
                                        vals['lot #'] == lot), 'measured'] = max_val
                        else:
                            vals.loc[(vals['param'] == prm) & (vals['device #'] == dev) & (vals['chip #'] == chp) & (
                                        vals['lot #'] == lot), 'measured'] -= max_val

        return vals

    def rename_series(measd, prm, new_name, device):
        new_df = measd.copy()
        new_df.loc[new_df['param'] == prm, 'device #'] = device
        new_df.loc[new_df['param'] == prm, 'param'] = new_name
        return new_df


    study_three = False
    if study_three:
        # Import the data from a JSON file
        h_data = gracefall.load_gerabaldi_report('C:/Users/Ian Hill/PycharmProjects/hill-scripts/data_files/idfbcamp_htol_lt_meas.json')
        h_data = h_data['Measurements']
        # Remove the measurement time that was only conducted for one of the lots
        h_data.drop(h_data[h_data['time'] == 3034800].index, inplace=True)
        h_data.drop(h_data[h_data['time'] == 3034860].index, inplace=True)
        h_data.drop(h_data[h_data['time'] == 3034920].index, inplace=True)
        # Rename direct measurement sensors
        h_data = rename_series(h_data, prm='pbti_vth_direct', new_name='pbti_vth', device=4)
        h_data = rename_series(h_data, prm='nbti_vth_direct', new_name='nbti_vth', device=4)
        h_data = rename_series(h_data, prm='nhci_vth_direct', new_name='nhci_vth', device=4)
        h_data = rename_series(h_data, prm='phci_vth_direct', new_name='phci_vth', device=4)
        # Drop all but the vth sensor measurements
        sensor_types = ['pbti_vth', 'nbti_vth', 'nhci_vth', 'phci_vth']
        h_data = h_data[h_data['param'].isin(sensor_types)]
        # Average the three measurements taken at each 100 hour mark
        h_data = avg_same_time_measurements(h_data)
        # Now extract the vth shift as the difference from the max value
        h_data = diff_from_max_val(h_data)
        # Don't care about which board the chip was on, just the chip IDs
        h_data['chip #'] = h_data['chip #'] + (2 * h_data['lot #'])

        deg_data = {}
        for t in jnp.linspace(0, 3_600_000, 11):
            hours = int(t / 3600)
            deg_data[f't{hours}'] = {}
            for sensor in sensor_types:
                vth = h_data.loc[(h_data['param'] == sensor) & (h_data['time'] == t)].drop(columns=['param', 'lot #', 'time'])
                vth = vth.sort_values(['chip #', 'device #'], ascending=[True, True])
                vth_array = jnp.array(vth['measured'].values).reshape((vth['chip #'].nunique(), vth['device #'].nunique()))
                deg_data[f't{hours}'][sensor] = vth_array

        # Now to work with the extracted measurement data
        print('ready')


if __name__ == '__main__':
    tddb_inference()
