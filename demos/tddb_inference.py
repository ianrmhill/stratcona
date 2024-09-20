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

import seaborn as sb
from matplotlib import pyplot as plt

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

        # E model for simulating some data to fit
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

        # Define the simulation test
        th = 130 + CELSIUS_TO_KELVIN
        test_130 = stratcona.ReliabilityTest({'e': {'lot': 1, 'chp': 5}}, {'e': {'temp': th, 'vg': 1.1}})

        am_e = stratcona.AnalysisManager(mb_e.build_model(), rng_seed=1295323)
        am_e.set_test_definition(test_130)
        ttfs = am_e.sim_test_measurements()
        fails = convert(ttfs['e'])

        # Frequentist analysis of the failure data
        mean, var = fails.mean(), fails.var()
        fit_info = reliability.Fitters.Fit_Weibull_2P(fails.tolist(), show_probability_plot=True)

        # Weibull CDF
        def CDF(x, k, L):
            return 1 - jnp.exp(- (x / L) ** k)

        # SPM for Weibull analysis
        mb_w = stratcona.SPMBuilder(mdl_name='weibull-2p')
        mb_w.add_hyperlatent('k_nom', dists.Normal, {'loc': 4.0, 'scale': 1.0}, transform=dists.transforms.SoftplusTransform())
        mb_w.add_hyperlatent('sc_nom', dists.Normal, {'loc': 1.9, 'scale': 0.4}, transform=dists.transforms.SoftplusTransform())
        var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
        mb_w.add_hyperlatent('k_dev', dists.Normal, {'loc': 0.3, 'scale': 0.1}, transform=var_tf)
        mb_w.add_hyperlatent('sc_dev', dists.Normal, {'loc': 0.4, 'scale': 0.1}, transform=var_tf)
        mb_w.add_hyperlatent('k_chp', dists.Normal, {'loc': 0.5, 'scale': 0.2}, transform=var_tf)
        mb_w.add_hyperlatent('sc_chp', dists.Normal, {'loc': 1.0, 'scale': 0.3}, transform=var_tf)
        mb_w.add_latent('k', nom='k_nom', dev='k_dev', chp='k_chp', lot=None)
        mb_w.add_latent('sc', nom='sc_nom', dev='sc_dev', chp='sc_chp', lot=None)
        #mb_w.add_latent('k', nom='k_nom', dev='k_dev', chp=None, lot=None)
        #mb_w.add_latent('sc', nom='sc_nom', dev='sc_dev', chp=None, lot=None)
        #mb_w.add_dependent('k_pos', lambda k: jnp.abs(k))
        #mb_w.add_dependent('sc_pos', lambda sc: jnp.abs(sc))
        mb_w.add_measured('ttf', dists.Weibull, {'concentration': 'k', 'scale': 'sc'}, 10)
        #mb_w.add_measured('ttf', dists.Normal, {'loc': 'k', 'scale': 'sc'}, 10)

        am_w = stratcona.AnalysisManager(mb_w.build_model(), rng_seed=92740189)

        test_130_sing = stratcona.ReliabilityTest({'e': {'lot': 1, 'chp': 1, 'ttf': 1}}, {'e': {'temp': th, 'vg': 1.1}})
        k1, k2, k3, k4 = rand.split(rand.key(428027234), 4)
        eval_sites = ['e_ttf_k_dev', 'e_ttf_sc_dev', 'e_k_chp', 'e_sc_chp',
                      'k_nom', 'sc_nom', 'k_dev', 'sc_dev', 'k_chp', 'sc_chp']

        prm_samples = am_w.relmdl.sample(k1, test_130_sing, 1000)
        ltnt_vals = {site: data for site, data in prm_samples.items() if site in eval_sites}
        pri_probs = jnp.exp(am_w.relmdl.logp(k2, test_130_sing, ltnt_vals, prm_samples))
        pri_probs = (pri_probs / (jnp.max(pri_probs) * 2)).flatten()

        x = jnp.logspace(-2, 1, 50)
        pri_fits = CDF(x, prm_samples['e_ttf_k'], prm_samples['e_ttf_sc'])

        am_w.do_inference(ttfs, test_130)

        prm_samples = am_w.relmdl.sample(k3, test_130_sing, 1000)
        ltnt_vals = {site: data for site, data in prm_samples.items() if site in eval_sites}
        pst_probs = jnp.exp(am_w.relmdl.logp(k4, test_130_sing, ltnt_vals, prm_samples))
        pst_probs = (pst_probs / (jnp.max(pst_probs) * 2)).flatten()

        pst_fits = CDF(x, prm_samples['e_ttf_k'], prm_samples['e_ttf_sc'])


        #def weibull(dev_count, fail_data=None):
        #    k = npyro.sample('k', dists.Normal(4.0, 2.0))
        #    sc = npyro.sample('sc', dists.Normal(1.9, 0.8))
        #    with npyro.plate('dev', dev_count):
        #        k_dev = npyro.sample('k_dev', dists.Normal(0.0, 0.05))
        #        sc_dev = npyro.sample('sc_dev', dists.Normal(0.0, 0.1))

        #        k_sum = k + k_dev
        #        sc_sum = sc + sc_dev
        #        fails = npyro.sample('fails', dists.Weibull(concentration=k, scale=sc), obs=fail_data)

        ## Inference the weibull model
        #kernel = npyro.infer.NUTS(weibull)
        #sampler = npyro.infer.MCMC(kernel, num_warmup=500, num_samples=3_000, num_chains=4)
        #rng_key = rand.key(6439578)
        #sampler.run(rng_key, dev_count=30, fail_data=fails, extra_fields=('potential_energy',))
        #samples = sampler.get_samples(group_by_chain=True)

        #convergence_stats = {}
        #for site in samples:
        #    convergence_stats[site] = {'ess': npyro.diagnostics.effective_sample_size(samples[site]),
        #                               'srhat': npyro.diagnostics.split_gelman_rubin(samples[site])}
        #extra_info = sampler.get_extra_fields()
        #diverging = extra_info['diverging'] if 'diverging' in extra_info else 0
        #diverging = jnp.sum(diverging)
        #print(convergence_stats)
        #print(f'Divergences: {diverging}')

        #k_new = stratcona.engine.inference.fit_dist_to_samples(dists.Normal, samples['k'])
        #sc_new = stratcona.engine.inference.fit_dist_to_samples(dists.Normal, samples['sc'])
        #print(f'k: {k_new}, sc: {sc_new}')

        # Generate some data series from the inferred model
        rng = rand.key(48408)
        k1, k2 = rand.split(rng)
        #prm_samples = am.relmdl.sample(k1, test_130, 100, keep_sites=['a_0_nom', 'e_aa_nom'])
        #pri_sample_probs = jnp.exp(am.relmdl.hyl_logp(k2, test_130, prm_samples))
        #pri_sample_probs = pri_sample_probs / (jnp.max(pri_sample_probs) * 2)
        #spm_temps = jnp.full((100, 13,), jnp.linspace(300, 420, 13)).T
        #pri_spm_vals = e_model_ttf(spm_temps, prm_samples['a_0_nom'], prm_samples['e_aa_nom'], BOLTZ_EV)
        #pri_spm_vals = pri_spm_vals.T

        # Get data for the weibull fit plot
        x = jnp.logspace(-2, 1, 50)
        fit_fails = CDF(x, fit_info.beta, fit_info.alpha)

        # Generate a Weibull plot!
        fails = jnp.sort(fails)
        n = len(fails)
        i = jnp.arange(1, n + 1)
        fail_order = (i - 0.5) / (n + 0.25)
        # Functions to correctly set up the axis scales
        ax_fwdy = lambda p: jnp.log(jnp.fmax(1e-20, -jnp.log(jnp.fmax(1e-20, 1 - p))))
        ax_bcky = lambda q: 1 - jnp.exp(-jnp.exp(q))
        ax_fwdx = lambda x: jnp.log(jnp.fmax(1e-20, x))
        ax_bckx = lambda y: jnp.exp(y)

        fig, p = plt.subplots(1, 1)
        p.plot(fails, fail_order, color='black', alpha=0.3, linestyle='', marker='.', markersize=4)
        p.plot(x, fit_fails, color='lightseagreen')

        # Plot the probabilistic fits
        for i in range(len(pri_fits)):
            p.plot(x, pri_fits[i].flatten(), alpha=float(pri_probs[i]), color='sienna')
        for i in range(len(pst_fits)):
            p.plot(x, pst_fits[i].flatten(), alpha=float(pst_probs[i]), color='rebeccapurple')

        p.set_xscale('function', functions=(ax_fwdx, ax_bckx))
        ln_min, ln_max = jnp.log(min(fails)), jnp.log(max(fails))
        lim_l = jnp.exp(ln_min - (0.05 * (ln_max - ln_min)))
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

            fig, p = plt.subplots(1, 1)
            # If the Bayesian model was inferred plot that first since it adds many series to the plot
            if infer_spm:
                for i in range(100):
                    p.plot(spm_temps[i], pri_spm_vals[i], color='burlywood', linestyle='-', alpha=float(pri_sample_probs[i]))
                    p.plot(spm_temps[i], spm_vals[i], color='royalblue', linestyle='-', alpha=float(sample_probs[i]))

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


    # Simulate four groups of two data sets, one with less observed data points, one with lots of points for each
    # of the four commonly used TDDB models


if __name__ == '__main__':
    tddb_inference()
