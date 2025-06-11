# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from pprint import pprint

import numpyro as npyro
import numpyro.distributions as dists
npyro.set_host_device_count(4)

import jax
import jax.numpy as jnp
import jax.random as rand

import time
from functools import partial
from scipy.optimize import curve_fit
from numpy.linalg import cholesky
import reliability
import pandas as pd

import seaborn as sb
from matplotlib import pyplot as plt

import gerabaldi
from gerabaldi.models import *

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stratcona

BOLTZ_EV = 8.617e-5
CELSIUS_TO_KELVIN = 273.15

SHOW_PLOTS = True


# Data manipulation helper function
def convert(vals):
    return vals['ttf'].flatten()


def tddb_inference():
    # Simulate TDDB distributions using all four classic TDDB models, then infer a model using both Bayesian and
    # frequentist techniques to compare. Only temperature acceleration considered.

    ### Define the simulation tests, one small, one large ###
    tl, tm, th = 105 + CELSIUS_TO_KELVIN, 125 + CELSIUS_TO_KELVIN, 145 + CELSIUS_TO_KELVIN
    test_s = stratcona.TestDef('small',
        {'el': {'lot': 1, 'chp': 1}, 'em': {'lot': 1, 'chp': 1}, 'eh': {'lot': 1, 'chp': 1}},
        {'el': {'temp': tl, 'vg': 1.1}, 'em': {'temp': tm, 'vg': 1.1}, 'eh': {'temp': th, 'vg': 1.1}})
    test_l = stratcona.TestDef('large',
        {'el': {'lot': 7, 'chp': 7}, 'em': {'lot': 7, 'chp': 7}, 'eh': {'lot': 7, 'chp': 7}},
        {'el': {'temp': tl, 'vg': 1.1}, 'em': {'temp': tm, 'vg': 1.1}, 'eh': {'temp': th, 'vg': 1.1}})

    fails_s = {'el': {}, 'em': {}, 'eh': {}}
    ttfs = {'s': {}, 'l': {}}
    fails_l = {'el': {}, 'em': {}, 'eh': {}}
    means, vars = {}, {}

    ### E model definition ###
    mb_e = stratcona.SPMBuilder(mdl_name='E-Model')

    def e_model_ttf(temp, a_o, e_aa, k):
        ttf_hours = 1e-5 * a_o * jnp.exp(e_aa / (k * temp))
        ttf_years = ttf_hours / 8760
        return ttf_years

    #mb_e.add_params(a_o_nom=3.8, a_o_dev=0.07, a_o_chp=0.01, a_o_lot=0.01)
    mb_e.add_params(a_o_nom=3.8, a_o_dev=0.02, a_o_chp=0.001, a_o_lot=0.004)
    mb_e.add_latent('a_o', nom='a_o_nom', dev='a_o_dev', chp='a_o_chp', lot='a_o_lot')
    #mb_e.add_params(e_aa_nom=0.70, e_aa_dev=0.02, e_aa_chp=0.01, e_aa_lot=0.01)
    mb_e.add_params(e_aa_nom=0.70, e_aa_dev=0.01, e_aa_chp=0.001, e_aa_lot=0.004)
    mb_e.add_latent('e_aa', nom='e_aa_nom', dev='e_aa_dev', chp='e_aa_chp', lot='e_aa_lot')

    mb_e.add_params(k=BOLTZ_EV, ttf_var=0.001)
    mb_e.add_intermediate('ttf_base', e_model_ttf)
    mb_e.add_observed('ttf', dists.Normal, {'loc': 'ttf_base', 'scale': 'ttf_var'}, 10)

    ### E model simulation ###
    means['e'], vars['e'] = {}, {}
    am_e = stratcona.AnalysisManager(mb_e.build_model(), rng_seed=1395323)
    am_e.set_test_definition(test_s)
    sim_tr = am_e.sim_test_measurements(rtrn_tr=False)
    ttfs['s']['e'] = {key: {'ttf': sim_tr[f'{key}_ttf']} for key in ['el', 'em', 'eh']}
    means['e']['el_s'], means['e']['em_s'], means['e']['eh_s'] = ttfs['s']['e']['el']['ttf'].mean(), ttfs['s']['e']['em']['ttf'].mean(), ttfs['s']['e']['eh']['ttf'].mean()
    vars['e']['el_s'], vars['e']['em_s'], vars['e']['eh_s'] = ttfs['s']['e']['el']['ttf'].var(), ttfs['s']['e']['em']['ttf'].var(), ttfs['s']['e']['eh']['ttf'].var()
    fails_s['el']['e'], fails_s['em']['e'], fails_s['eh']['e'] = convert(ttfs['s']['e']['el']), convert(ttfs['s']['e']['em']), convert(ttfs['s']['e']['eh'])
    am_e.set_test_definition(test_l)
    sim_tr = am_e.sim_test_measurements(rtrn_tr=False)
    ttfs['l']['e'] = {key: {'ttf': sim_tr[f'{key}_ttf']} for key in ['el', 'em', 'eh']}
    means['e']['el_l'], means['e']['em_l'], means['e']['eh_l'] = ttfs['l']['e']['el']['ttf'].mean(), ttfs['l']['e']['em']['ttf'].mean(), ttfs['l']['e']['eh']['ttf'].mean()
    vars['e']['el_l'], vars['e']['em_l'], vars['e']['eh_l'] = ttfs['l']['e']['el']['ttf'].var(), ttfs['l']['e']['em']['ttf'].var(), ttfs['l']['e']['eh']['ttf'].var()
    fails_l['el']['e'], fails_l['em']['e'], fails_l['eh']['e'] = convert(ttfs['l']['e']['el']), convert(ttfs['l']['e']['em']), convert(ttfs['l']['e']['eh'])

    ### 1/E model definition, reference paper: Temperature Acceleration of Time-Dependent Dielectric Breakdown: 10.1109/16.43668
    mb_inv_e = stratcona.SPMBuilder(mdl_name='Inverse-E-Model')

    def inv_e_model_ttf(temp, t_o, e_b, e_ox, g_o, delta, k):
        tau_o = t_o * jnp.exp((-(e_b * 0.001) / k) * ((1 / temp) - (1 / 300)))
        g = g_o * (1 + (((delta * 0.1) / k) * ((1 / temp) - (1 / 300))))
        ttf_hours = 1e6 * tau_o * jnp.exp(g / e_ox)
        ttf_years = ttf_hours / 8760
        return ttf_years

    mb_inv_e.add_params(t_o_nom=4.3, t_o_dev=0.4, t_o_chp=0.25, t_o_lot=0.23)
    mb_inv_e.add_latent('t_o', nom='t_o_nom', dev='t_o_dev', chp='t_o_chp', lot='t_o_lot')
    mb_inv_e.add_params(g_o_nom=2.5, g_o_dev=0.22, g_o_chp=0.11, g_o_lot=0.01)
    mb_inv_e.add_latent('g_o', nom='g_o_nom', dev='g_o_dev', chp='g_o_chp', lot='g_o_lot')
    mb_inv_e.add_params(e_b_nom=1.7, e_b_dev=0.07, e_b_chp=0.01, e_b_lot=0.01)
    mb_inv_e.add_latent('e_b', nom='e_b_nom', dev='e_b_dev', chp='e_b_chp', lot='e_b_lot')
    mb_inv_e.add_params(delta_nom=7.5, delta_dev=0.22, delta_chp=0.11, delta_lot=0.01)
    mb_inv_e.add_latent('delta', nom='delta_nom', dev='delta_dev', chp='delta_chp', lot='delta_lot')
    mb_inv_e.add_params(e_ox_nom=3.2, e_ox_dev=0.12, e_ox_chp=0.01, e_ox_lot=0.01)
    mb_inv_e.add_latent('e_ox', nom='e_ox_nom', dev='e_ox_dev', chp='e_ox_chp', lot='e_ox_lot')

    mb_inv_e.add_params(k=BOLTZ_EV, ttf_var=0.001)
    mb_inv_e.add_intermediate('inv_e_ttf', inv_e_model_ttf)
    mb_inv_e.add_observed('ttf', dists.Normal, {'loc': 'inv_e_ttf', 'scale': 'ttf_var'}, 10)

    ### 1/E model simulation
    means['ie'], vars['ie'] = {}, {}
    am_inv_e = stratcona.AnalysisManager(mb_inv_e.build_model(), rng_seed=3229823)
    am_inv_e.set_test_definition(test_s)
    sim_tr = am_inv_e.sim_test_measurements(rtrn_tr=False)
    ttfs['s']['ie'] = {key: {'ttf': sim_tr[f'{key}_ttf']} for key in ['el', 'em', 'eh']}
    means['ie']['el_s'], means['ie']['em_s'], means['ie']['eh_s'] = ttfs['s']['ie']['el']['ttf'].mean(), ttfs['s']['ie']['em']['ttf'].mean(), ttfs['s']['ie']['eh']['ttf'].mean()
    vars['ie']['el_s'], vars['ie']['em_s'], vars['ie']['eh_s'] = ttfs['s']['ie']['el']['ttf'].var(), ttfs['s']['ie']['em']['ttf'].var(), ttfs['s']['ie']['eh']['ttf'].var()
    fails_s['el']['ie'], fails_s['em']['ie'], fails_s['eh']['ie'] = convert(ttfs['s']['ie']['el']), convert(ttfs['s']['ie']['em']), convert(ttfs['s']['ie']['eh'])
    am_inv_e.set_test_definition(test_l)
    sim_tr = am_inv_e.sim_test_measurements(rtrn_tr=False)
    ttfs['l']['ie'] = {key: {'ttf': sim_tr[f'{key}_ttf']} for key in ['el', 'em', 'eh']}
    means['ie']['el_l'], means['ie']['em_l'], means['ie']['eh_l'] = ttfs['l']['ie']['el']['ttf'].mean(), ttfs['l']['ie']['em']['ttf'].mean(), ttfs['l']['ie']['eh']['ttf'].mean()
    vars['ie']['el_l'], vars['ie']['em_l'], vars['ie']['eh_l'] = ttfs['l']['ie']['el']['ttf'].var(), ttfs['l']['ie']['em']['ttf'].var(), ttfs['l']['ie']['eh']['ttf'].var()
    fails_l['el']['ie'], fails_l['em']['ie'], fails_l['eh']['ie'] = convert(ttfs['l']['ie']['el']), convert(ttfs['l']['ie']['em']), convert(ttfs['l']['ie']['eh'])

    # NOTE: With constant voltage, V model ends up having identical form to the E model just with a prescale, so
    #       it is not used here

    ### Power law model definition, reference: Interplay of voltage and temperature acceleration of oxide breakdown for
    # ultra-thin gate oxides: https://doi.org/10.1016/S0038-1101(02)00151-X
    mb_p = stratcona.SPMBuilder(mdl_name='Power-Law-Model')

    def p_model_ttf(temp, t_o, n, a, b, vg):
        ttf_hours = 1e0 * t_o * (vg ** -n) * jnp.exp(((a * 1000) / temp) + ((b * 100000) / (temp ** 2)))
        ttf_years = ttf_hours / 8760
        return ttf_years

    mb_p.add_params(t_o_nom=1.54, t_o_dev=0.2, t_o_chp=0.16, t_o_lot=0.15)
    mb_p.add_latent('t_o', nom='t_o_nom', dev='t_o_dev', chp='t_o_chp', lot='t_o_lot')
    mb_p.add_params(n_nom=2.4, n_dev=0.01, n_chp=0.01, n_lot=0.01)
    mb_p.add_latent('n', nom='n_nom', dev='n_dev', chp='n_chp', lot='n_lot')
    mb_p.add_params(a_nom=3.1, a_dev=0.21, a_chp=0.11, a_lot=0.11)
    mb_p.add_latent('a', nom='a_nom', dev='a_dev', chp='a_chp', lot='a_lot')
    mb_p.add_params(b_nom=4.4, b_dev=0.31, b_chp=0.21, b_lot=0.11)
    mb_p.add_latent('b', nom='b_nom', dev='b_dev', chp='b_chp', lot='b_lot')

    mb_p.add_params(k=BOLTZ_EV, ttf_var=0.001)
    mb_p.add_intermediate('ttf_base', p_model_ttf)
    mb_p.add_observed('ttf', dists.Normal, {'loc': 'ttf_base', 'scale': 'ttf_var'}, 10)

    ### Power law model simulation ###
    means['p'], vars['p'] = {}, {}
    am_p = stratcona.AnalysisManager(mb_p.build_model(), rng_seed=1299323)
    am_p.set_test_definition(test_s)
    sim_tr = am_p.sim_test_measurements(rtrn_tr=False)
    ttfs['s']['p'] = {key: {'ttf': sim_tr[f'{key}_ttf']} for key in ['el', 'em', 'eh']}
    means['p']['el_s'], means['p']['em_s'], means['p']['eh_s'] = ttfs['s']['p']['el']['ttf'].mean(), ttfs['s']['p']['em']['ttf'].mean(), ttfs['s']['p']['eh']['ttf'].mean()
    vars['p']['el_s'], vars['p']['em_s'], vars['p']['eh_s'] = ttfs['s']['p']['el']['ttf'].var(), ttfs['s']['p']['em']['ttf'].var(), ttfs['s']['p']['eh']['ttf'].var()
    fails_s['el']['p'], fails_s['em']['p'], fails_s['eh']['p'] = convert(ttfs['s']['p']['el']), convert(ttfs['s']['p']['em']), convert(ttfs['s']['p']['eh'])
    am_p.set_test_definition(test_l)
    sim_tr = am_p.sim_test_measurements(rtrn_tr=False)
    ttfs['l']['p'] = {key: {'ttf': sim_tr[f'{key}_ttf']} for key in ['el', 'em', 'eh']}
    means['p']['el_l'], means['p']['em_l'], means['p']['eh_l'] = ttfs['l']['p']['el']['ttf'].mean(), ttfs['l']['p']['em']['ttf'].mean(), ttfs['l']['p']['eh']['ttf'].mean()
    vars['p']['el_l'], vars['p']['em_l'], vars['p']['eh_l'] = ttfs['l']['p']['el']['ttf'].var(), ttfs['l']['p']['em']['ttf'].var(), ttfs['l']['p']['eh']['ttf'].var()
    fails_l['el']['p'], fails_l['em']['p'], fails_l['eh']['p'] = convert(ttfs['l']['p']['el']), convert(ttfs['l']['p']['em']), convert(ttfs['l']['p']['eh'])

    ### Simulating data complete
    for k in means:
        print(f"Means of {k}: low s: {means[k]['el_s']}, med s: {means[k]['em_s']}, hi s: {means[k]['eh_s']}, low l: {means[k]['el_l']}, med l: {means[k]['em_l']}, hi l: {means[k]['eh_l']}")
        print(f"Variances of {k}: low s: {vars[k]['el_s']}, med s: {vars[k]['em_s']}, hi s: {vars[k]['eh_s']}, low l: {vars[k]['el_l']}, med l: {vars[k]['em_l']}, hi l: {vars[k]['eh_l']}")

    # Determine the probability of the small test data sample under the simulation model, too low for plotting though
    #nom_ttfs = {'el': 4.3, 'em': 2.1, 'eh': 1.2}
    #sample_sites, val_map = [], {}
    #for temp in ['el', 'em', 'eh']:
    #    for ltnt in ['a_o', 'e_aa']:
    #        for lyr in ['dev', 'chp', 'lot']:
    #            if lyr == 'dev':
    #                sample_sites.append(f'{temp}_ttf_{ltnt}_{lyr}_ls')
    #                val_map[f'{temp}_ttf_{ltnt}_{lyr}_ls'] = 0.0
    #            else:
    #                sample_sites.append(f'{temp}_{ltnt}_{lyr}_ls')
    #                val_map[f'{temp}_{ltnt}_{lyr}_ls'] = 0.0
    #    sample_sites.append(f'{temp}_ttf')
    #    val_map[f'{temp}_ttf'] = nom_ttfs[temp]

    #am_e.set_test_definition(test_s)
    #sim_tr = am_e.sim_test_measurements(rtrn_tr=False)
    #k1, k2 = rand.split(rand.key(7932854))
    #nom_vals = {key: jnp.full_like(sim_tr[key], val_map[key]) for key in sample_sites}
    #mean_prob = jnp.exp(am_e.relmdl.logp(k1, test_s, nom_vals, sim_tr))
    #sim_vals = {key: sim_tr[key] for key in sample_sites}
    #sim_prob = jnp.exp(am_e.relmdl.logp(k2, test_s, sim_vals, sim_tr))
    #normd_probs = sim_prob / mean_prob

    ### Plot the simulated data ###
    sb.set_theme(style='ticks', font='Times New Roman')
    sb.set_context('notebook')
    model_colours = ['mediumblue', 'darkviolet', 'lightseagreen']

    fig, p = plt.subplots(1, 3)
    sb.stripplot(data=fails_s['el'], ax=p[0], palette=model_colours, orient='h')
    p[0].set_title('T=105C')
    sb.stripplot(data=fails_s['em'], ax=p[1], palette=model_colours, orient='h')
    p[1].set_title('T=125C')
    sb.stripplot(data=fails_s['eh'], ax=p[2], palette=model_colours, orient='h')
    p[2].set_title('T=145C')
    for i in range(len(p)):
        p[i].set_yticks([0, 1, 2], labels=['E Model', '1/E Model', 'Power Law Model'])
        p[i].set_ylabel(None)
        p[i].set_xlabel('Time to Failure (years)')
        p[i].set_xscale('log')
    fig.subplots_adjust(hspace=10)

    fig, p = plt.subplots(1, 3)
    sb.stripplot(data=fails_l['el'], ax=p[0], palette=model_colours, orient='h')
    p[0].set_title('T=105C')
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
    #plt.show()

    ### Define the Power Law SPM for inference ###
    mb = stratcona.SPMBuilder(mdl_name='Power-Model')

    #t_0_nom_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.0001)])
    mb.add_hyperlatent('t_o_nom', dists.LogNormal, {'loc': 2.34, 'scale': 1.9}, transform=dists.transforms.AffineTransform(0, 0.0001))
    #t_0_var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.00001)])
    mb.add_hyperlatent('t_o_var', dists.LogNormal, {'loc': 2.5, 'scale': 1.4}, transform=dists.transforms.AffineTransform(0, 0.00001))

    mb.add_hyperlatent('n_nom', dists.Normal, {'loc': 2.5, 'scale': 0.01})
    #n_var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    #mb.add_hyperlatent('n_var', dists.Normal, {'loc': 2.5, 'scale': 1.3}, transform=n_var_tf)

    mb.add_hyperlatent('a_nom', dists.Normal, {'loc': 6.4, 'scale': 0.4})
    n_var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb.add_hyperlatent('a_var', dists.Normal, {'loc': 1.5, 'scale': 1.3}, transform=n_var_tf)

    mb.add_hyperlatent('b_nom', dists.Normal, {'loc': 2.9, 'scale': 0.4})
    n_var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb.add_hyperlatent('b_var', dists.Normal, {'loc': 1.5, 'scale': 1.3}, transform=n_var_tf)

    #ttf_hours = 1e0 * t_o * (vg ** -n) * jnp.exp(((a * 1000) / temp) + ((b * 100000) / (temp ** 2)))
    mb.add_latent('t_o', nom='t_o_nom', dev='t_o_var', chp=None, lot=None)
    mb.add_latent('n', nom='n_nom', dev=None, chp=None, lot=None)
    mb.add_latent('a', nom='a_nom', dev='a_var', chp=None, lot=None)
    mb.add_latent('b', nom='b_nom', dev='b_var', chp=None, lot=None)

    mb.add_params(k=BOLTZ_EV, ttf_var=2.4)
    mb.add_intermediate('ttf_base', p_model_ttf)
    mb.add_observed('ttf', dists.Normal, {'loc': 'ttf_base', 'scale': 'ttf_var'}, 10)
    mb.add_fail_criterion('field_life', lambda ttf: ttf)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=72864823)
    am.set_field_use_conditions({'temp': 55 + CELSIUS_TO_KELVIN})

    ### Sample some curves from the prior predictive ###
    num_curves = 1000
    test = stratcona.TestDef('single',
        {'e': {'lot': 1, 'chp': 1}},
        {'e': {'temp': 300, 'vg': 1.1}, 'em': {'temp': tm, 'vg': 1.1}, 'eh': {'temp': th, 'vg': 1.1}})
    rng = rand.key(48408)
    k1, k2 = rand.split(rng)
    prm_samples = am.relmdl.sample(k1, test, (num_curves,), keep_sites=['t_o_nom', 'n_nom', 'a_nom', 'b_nom'])
    pri_sample_probs = jnp.exp(am.relmdl.logp(k2, test, prm_samples, None, (num_curves,)))
    pri_sample_probs = pri_sample_probs / (jnp.max(pri_sample_probs) * 2)
    spm_temps = jnp.full((num_curves, 13,), jnp.linspace(300, 420, 13)).T
    pri_spm_vals = p_model_ttf(spm_temps, prm_samples['t_o_nom'], prm_samples['n_nom'], prm_samples['a_nom'], prm_samples['b_nom'], 1.1)
    pri_spm_vals = pri_spm_vals.T

    ### Generate predictive distribution at field use for prior ###
    am.set_field_use_conditions({'temp': 330, 'vg': 1.1})
    am.relreq = stratcona.ReliabilityRequirement(stratcona.engine.metrics.qx_lbci_l, 95, 10)
    pri_lbci = am.evaluate_reliability('ttf_ttf_base')
    am.relreq = stratcona.ReliabilityRequirement(stratcona.engine.metrics.qx_hdcr_l, 95, 10)
    pri_hdcr = am.evaluate_reliability('ttf_ttf_base')
    am.test = am.field_test
    pri_dist = am.sim_test_meas_new(num=(300_000,))
    pri_dist = pri_dist['field_ttf'].flatten()

    ### Evaluate the prior interpretable entropy ###
    ENTROPY_SAMPLES = 30_000
    def lp_f(vals, site, key, test):
        return am.relmdl.logp(rng_key=key, test=test, site_vals={site: vals}, conditional=None, dims=(len(vals),))
    k1, k2, k3, k4 = rand.split(rand.key(84357345763), 4)
    hyls = ('t_o_nom', 't_o_var', 'n_nom', 'a_nom', 'a_var', 'b_nom', 'b_var')
    hyl_samples = am.relmdl.sample_new(k1, am.test.dims, am.test.conds, (ENTROPY_SAMPLES,), keep_sites=hyls)
    pri_entropy = {}
    for hyl in hyls:
        pri_entropy[hyl] = stratcona.engine.bed.entropy(
            hyl_samples[hyl], partial(lp_f, site=hyl, test=am.test, key=k1), limiting_density_range=(-838.8608, 838.8607))
    print(f'Prior interpretable entropy: {pri_entropy}')

    ### Inference the SPM ###
    am.set_test_definition(test_s)
    start_time = time.time()

    am.do_inference(ttfs['s']['e'])
    #am.relmdl.hyl_beliefs['t_o_nom']['loc'] = 4
    #am.relmdl.hyl_beliefs['t_o_var']['loc'] = 1.5
    #am.relmdl.hyl_beliefs['t_o_var']['scale'] = 0.5

    print(f'Inference time taken: {time.time() - start_time}')
    pprint(am.relmdl.hyl_beliefs)
    jax.clear_caches()

    t_o_infd = am.relmdl.hyl_beliefs['t_o_nom']['loc']
    n_infd = am.relmdl.hyl_beliefs['n_nom']['loc']
    a_infd = am.relmdl.hyl_beliefs['a_nom']['loc']
    b_infd = am.relmdl.hyl_beliefs['b_nom']['loc']
    ttf_l = p_model_ttf(temp=tl, t_o=t_o_infd, n=n_infd, a=a_infd, b=b_infd, vg=1.1)
    ttf_m = p_model_ttf(temp=tm, t_o=t_o_infd, n=n_infd, a=a_infd, b=b_infd, vg=1.1)
    ttf_h = p_model_ttf(temp=th, t_o=t_o_infd, n=n_infd, a=a_infd, b=b_infd, vg=1.1)

    mean_temps = jnp.linspace(300, 420, 13)
    mean_infd = p_model_ttf(temp=mean_temps, t_o=t_o_infd, n=n_infd, a=a_infd, b=b_infd, vg=1.1)
    print(f'Inference model predicted MTTF: 105C={ttf_l}, 125C={ttf_m}, 145C={ttf_h}')

    # Sample curves from the posterior predictive
    rng, kx = rand.split(rand.key(27498))
    prm_samples = am.relmdl.sample(rng, test, (num_curves,), keep_sites=['t_o_nom', 'n_nom', 'a_nom', 'b_nom'])
    sample_probs = jnp.exp(am.relmdl.logp(k2, test, prm_samples, conditional=None, dims=(num_curves,)))
    sample_probs = sample_probs / (jnp.max(sample_probs) * 2)
    spm_vals = p_model_ttf(spm_temps, prm_samples['t_o_nom'], prm_samples['n_nom'], prm_samples['a_nom'], prm_samples['b_nom'], 1.1)
    spm_vals, spm_temps = spm_vals.T, spm_temps.T

    ### Evaluate the posterior entropy ###
    hyl_samples = am.relmdl.sample_new(k3, am.test.dims, am.test.conds, (ENTROPY_SAMPLES,), keep_sites=hyls)
    pst_entropy = {}
    for hyl in hyls:
        pst_entropy[hyl] = stratcona.engine.bed.entropy(
            hyl_samples[hyl], partial(lp_f, site=hyl, test=am.test, key=k3), limiting_density_range=(-838.8608, 838.8607))
    print(f'Posterior interpretable entropy: {pst_entropy}')


    temps_s = jnp.repeat(jnp.array([tl, tm, th]), 10)
    temps_l = jnp.repeat(jnp.array([tl, tm, th]), 10 * 7 * 7)

    funcs = {}
    funcs['e'] = partial(e_model_ttf, k=BOLTZ_EV)
    funcs['ie'] = partial(inv_e_model_ttf, k=BOLTZ_EV)
    funcs['p'] = partial(p_model_ttf, vg=1.1)

    # Fit the models with MLE, generate confidence intervals
    fits = {'s': {}, 'l': {}}
    #for mdl in ['e', 'ie', 'p']:
    for mdl in ['e', 'p']:
        fits['s'][mdl], fits['l'][mdl] = {}, {}
        #for sim in ['e', 'ie', 'p']:
        for sim in ['e']:
            s_fails = jnp.concatenate((convert(ttfs['s'][sim]['el']), convert(ttfs['s'][sim]['em']), convert(ttfs['s'][sim]['eh'])))
            l_fails = jnp.concatenate((convert(ttfs['l'][sim]['el']), convert(ttfs['l'][sim]['em']), convert(ttfs['l'][sim]['eh'])))
            try:
                sigma = 2.4 # 5.76
                fits['s'][mdl][sim] = curve_fit(funcs[mdl], xdata=temps_s, ydata=s_fails)#, sigma=sigma, absolute_sigma=True)
                fits['l'][mdl][sim] = curve_fit(funcs[mdl], xdata=temps_l, ydata=l_fails)
            except RuntimeError as e:
                print(e)
                fits['s'][mdl][sim] = None
                fits['l'][mdl][sim] = None
    print(fits)

    ### Now for plotting
    y_ttfs = {'s': {}, 'l': {}}
    x_temps = jnp.linspace(300, 420, 13)
    y_ttfs['s']['e'] = funcs['e'](x_temps, *fits['s']['e']['e'][0])
    y_ttfs['l']['e'] = funcs['e'](x_temps, *fits['l']['e']['e'][0])
    fit_prms = fits['s']['e']['e']
    e_sampler = dists.MultivariateStudentT(8, fit_prms[0], cholesky(fit_prms[1]))
    rng_key = rand.key(23492)
    #print(e_sampler.sample(rng_key, (10,)))

    #y_ttfs['s']['ie'] = funcs['ie'](x_temps, *fits['s']['ie']['e'][0])
    #y_ttfs['l']['ie'] = funcs['ie'](x_temps, *fits['l']['ie']['e'][0])
    #y_ttfs['s']['p'] = funcs['p'](x_temps, *fits['s']['p']['e'][0])
    #y_ttfs['l']['p'] = funcs['p'](x_temps, *fits['l']['p']['e'][0])

    sb.set_context('notebook')
    fig, p = plt.subplots(1, 1)
    p.grid()

    for i in range(num_curves):
        lbl = 'Prior predictive distribution' if i == 0 else None
        p.plot(spm_temps[i], pri_spm_vals[i], color='skyblue', linestyle='-', alpha=float(pri_sample_probs[i]), label=lbl)
        lbl = 'Posterior predictive distribution' if i == 0 else None
        p.plot(spm_temps[i], spm_vals[i], color='darkblue', linestyle='-', alpha=float(sample_probs[i]), label=lbl)
    #p.plot(mean_temps, mean_infd, color='navy')

    # Plot the large set of simulated data points to give an idea of variability
    l_fails = jnp.concatenate((convert(ttfs['l']['e']['el']), convert(ttfs['l']['e']['em']), convert(ttfs['l']['e']['eh'])))
    p.plot(temps_l, l_fails, color='sienna', alpha=1, linestyle='', marker='.', markersize=4)
    # Plot the curve used to generate the observed data
    p.plot(x_temps, funcs['e'](x_temps, 3.8, 0.7), color='sienna', linewidth=3, label='Simulation model "ground truth"')
    # Plot the small simulated data set
    s_fails = jnp.concatenate((convert(ttfs['s']['e']['el']), convert(ttfs['s']['e']['em']), convert(ttfs['s']['e']['eh'])))
    p.plot(temps_s, s_fails, color='darkorange', linestyle='', marker='.', markersize=10, label='Simulated test data')


    # Plot the frequentist fits to the data
    mdl_clrs = ['deeppink']
    for i, k in enumerate(['e']):
        lbl = 'Power law model fit' if k == 'p' else 'Frequentist MLE fit to test data'
        p.plot(x_temps, y_ttfs['s'][k], color=mdl_clrs[i], linestyle='--', linewidth=2, label=lbl)
        #p.plot(x_temps, y_ttfs['l'][k], color=mdl_clrs[i], linestyle='--', linewidth=2)

    p.set_xlabel('Temperature (K)')
    p.set_xlim(300, 420)
    p.set_ylim(1e-1, 1e4)
    p.set_ylabel('Time to Failure (years)')
    p.set_yscale('log')

    hndl, lbls = p.get_legend_handles_labels()
    lgnd_order = [3, 2, 4, 0, 1]
    leg = p.legend([hndl[i] for i in lgnd_order], [lbls[i] for i in lgnd_order], loc='lower left')
    for lbl in leg.legend_handles:
        lbl.set_alpha(1)

    p.annotate(f'Historical knowledge encoded in Bayesian approach\nreduces underestimation of temperature dependence\ngiven unlucky test data',
               (312, 1e3), (330, 1e3),
               arrowprops={'arrowstyle': 'simple', 'color': 'black'})

    ### Generate predictive plot showing the metrics, compare to frequentist predictions and true model ###
    am.set_field_use_conditions({'temp': 330, 'vg': 1.1})
    am.relreq = stratcona.ReliabilityRequirement(stratcona.engine.metrics.qx_lbci_l, 95, 10)
    post_lbci = am.evaluate_reliability('ttf_ttf_base')
    am.relreq = stratcona.ReliabilityRequirement(stratcona.engine.metrics.qx_hdcr_l, 95, 10)
    post_hdcr = am.evaluate_reliability('ttf_ttf_base')
    am.test = am.field_test
    post_dist = am.sim_test_meas_new(num=(300_000,))
    post_dist = post_dist['field_ttf'].flatten()
    # Generate 'true' samples
    am_e.test = am.field_test
    true_dist = am_e.sim_test_meas_new(num=(300_000,))['field_ttf'].flatten()
    # Generate mean predicted value from frequentist fit
    mle_mean = fits['s']['e']['e'][0]
    mle_pred_mean = e_model_ttf(330, mle_mean[0], mle_mean[1], BOLTZ_EV)
    # Figure out the 95% prediction interval
    ci_std_err = jnp.sqrt(jnp.diag(fits['s']['e']['e'][1]))
    dlf_da = jax.jacfwd(e_model_ttf, 1)
    dlf_de = jax.jacfwd(e_model_ttf, 2)
    def var_lf(a, a_se, e, e_se, cov_ae, temp):
        return (((dlf_da(temp, a, e, BOLTZ_EV) ** 2) * (a_se ** 2)) +
                ((dlf_de(temp, a, e, BOLTZ_EV) ** 2) * (e_se ** 2)) +
                (2 * dlf_da(temp, a, e, BOLTZ_EV) * dlf_de(temp, a, e, BOLTZ_EV) * cov_ae))
    z = 1.96
    e_fit_prms = fits['s']['e']['e']
    ae_se = e_fit_prms[1][0, 1]
    a, a_se, e, e_se = mle_mean[0], ci_std_err[0], mle_mean[1], ci_std_err[1]
    lf_lower = e_model_ttf(330, a, e, BOLTZ_EV) - (z * jnp.sqrt(var_lf(a, a_se, e, e_se, ae_se, 330)))
    lf_upper = e_model_ttf(330, a, e, BOLTZ_EV) + (z * jnp.sqrt(var_lf(a, a_se, e, e_se, ae_se, 330)))
    if lf_lower < 0:
        lf_lower = 0.01

    sb.set_context('notebook')
    fig, p = plt.subplots(1, 1)
    #p.grid()
    p.set_xlim(0, 11.513)
    p.set_xlabel('Failure Time (years)')
    p.set_xticks(ticks=[0, 2.303, 4.605, 6.908, 9.21, 11.513],
                 labels=["1", "10", "10^2", "10^3", "10^4", "10^5"])
    p.set_ylabel('Probability Density (At field use - 330K)')
    p.set_yticks([])

    p.hist(jnp.log(true_dist[true_dist >= 0.01]), 500, density=True, alpha=0.9, color='goldenrod', histtype='stepfilled', label='Simulated true lifespan')
    p.hist(jnp.log(pri_dist[pri_dist >= 0.01]), 500, density=True, alpha=0.8, color='skyblue', histtype='stepfilled', label='Prior predicted lifespan')
    p.hist(jnp.log(post_dist[post_dist >= 0.01]), 500, density=True, alpha=0.75, color='darkblue', histtype='stepfilled', label='Posterior predicted lifespan')

    p.axvline(float(jnp.log(mle_pred_mean)), 0, 1, color='deeppink', linestyle='dashed', label=f'MLE fit predicted mean: {round(float(mle_pred_mean), 2)}')
    p.axvspan(float(jnp.log(lf_lower)), float(jnp.log(lf_upper)), color='deeppink', linestyle='dashed', alpha=0.2,
              #label=f'MLE 95% CI: [{round(float(lf_lower), 2)}, {round(float(lf_upper), 2)}]')
              label = f'MLE 95% CI max: {round(float(lf_upper), 2)}')

    p.axvline(float(jnp.log(post_lbci)), 0, 1, color='darkviolet', linestyle='dashed',
              label=f'Q95%-LBCI: {round(float(post_lbci), 2)}')
    label_made = False
    for interval in post_hdcr:
        label = f'Q95%-HDCR: [{round(float(post_hdcr[0][0]), 2)}, {round(float(post_hdcr[0][1]), 2)}]' if not label_made else None
        if label is not None:
            label_made = True
        p.axvspan(float(jnp.log(interval[0])), float(jnp.log(interval[1])), color='darkgreen', linestyle='dashed', alpha=0.2,
                  label=label)

    # Special legend handling if needed
    #hndl, lbls = p.get_legend_handles_labels()
    #lgnd1 = pltlines.Line2D([0], [0], color='black', linestyle='--')
    #lgnd2 = pltlines.Line2D([0], [0], color='black', linestyle='', marker='.')
    #hndl.insert(0, lgnd1)
    #lbls.insert(0, 'Lot simulation distributions')
    #hndl.insert(0, lgnd2)
    #lbls.insert(0, 'Simulated data, coloured by lot')

    ## Add the custom legend
    #leg = p.legend(hndl, lbls, loc='lower right')
    #for lbl in leg.legend_handles:
    #    lbl.set_alpha(1)

    p.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    tddb_inference()
