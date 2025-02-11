# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from pprint import pprint

import numpyro as npyro
import numpyro.distributions as dists
npyro.set_host_device_count(4)

import jax.numpy as jnp
import jax.random as rand

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


def tddb_inference():

    # Data manipulation helper function
    def convert(vals):
        return vals['ttf'].flatten()

    # Simulate TDDB distributions using all four classic TDDB models, then infer a model using both Bayesian and
    # frequentist techniques to compare. Only temperature acceleration considered.
    # Define the simulation test
    tl, tm, th = 85 + CELSIUS_TO_KELVIN, 125 + CELSIUS_TO_KELVIN, 145 + CELSIUS_TO_KELVIN
    test_s = stratcona.ReliabilityTest(
        {'el': {'lot': 1, 'chp': 1}, 'em': {'lot': 1, 'chp': 1}, 'eh': {'lot': 1, 'chp': 1}},
        {'el': {'temp': tl, 'vg': 1.1}, 'em': {'temp': tm, 'vg': 1.1}, 'eh': {'temp': th, 'vg': 1.1}})
    test_l = stratcona.ReliabilityTest(
        {'el': {'lot': 7, 'chp': 7}, 'em': {'lot': 7, 'chp': 7}, 'eh': {'lot': 7, 'chp': 7}},
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
    mb_e.add_intermediate('ttf_base', e_model_ttf)
    mb_e.add_observed('ttf', dists.Normal, {'loc': 'ttf_base', 'scale': 'ttf_var'}, 10)

    means['e'], vars['e'] = {}, {}
    am_e = stratcona.AnalysisManager(mb_e.build_model(), rng_seed=1295323)
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
    mb_inv_e.add_intermediate('inv_e_ttf', inv_e_model_ttf)
    mb_inv_e.add_observed('ttf', dists.Normal, {'loc': 'inv_e_ttf', 'scale': 'ttf_var'}, 10)

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
    mb_p.add_intermediate('ttf_base', p_model_ttf)
    mb_p.add_observed('ttf', dists.Normal, {'loc': 'ttf_base', 'scale': 'ttf_var'}, 10)

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

    # Simulating data complete
    for k in means:
        print(f"Means of {k}: low s: {means[k]['el_s']}, med s: {means[k]['em_s']}, hi s: {means[k]['eh_s']}, low l: {means[k]['el_l']}, med l: {means[k]['em_l']}, hi l: {means[k]['eh_l']}")
        print(f"Variances of {k}: low s: {vars[k]['el_s']}, med s: {vars[k]['em_s']}, hi s: {vars[k]['eh_s']}, low l: {vars[k]['el_l']}, med l: {vars[k]['em_l']}, hi l: {vars[k]['eh_l']}")

    # Determine the probability of the small test data sample under the simulation model, too low for plotting though
    nom_ttfs = {'el': 4.3, 'em': 2.1, 'eh': 1.2}
    sample_sites, val_map = [], {}
    for temp in ['el', 'em', 'eh']:
        for ltnt in ['a_o', 'e_aa']:
            for lyr in ['dev', 'chp', 'lot']:
                if lyr == 'dev':
                    sample_sites.append(f'{temp}_ttf_{ltnt}_{lyr}_ls')
                    val_map[f'{temp}_ttf_{ltnt}_{lyr}_ls'] = 0.0
                else:
                    sample_sites.append(f'{temp}_{ltnt}_{lyr}_ls')
                    val_map[f'{temp}_{ltnt}_{lyr}_ls'] = 0.0
        sample_sites.append(f'{temp}_ttf')
        val_map[f'{temp}_ttf'] = nom_ttfs[temp]

    am_e.set_test_definition(test_s)
    sim_tr = am_e.sim_test_measurements(rtrn_tr=False)
    k1, k2 = rand.split(rand.key(7932854))
    nom_vals = {key: jnp.full_like(sim_tr[key], val_map[key]) for key in sample_sites}
    mean_prob = jnp.exp(am_e.relmdl.logp(k1, test_s, nom_vals, sim_tr))
    sim_vals = {key: sim_tr[key] for key in sample_sites}
    sim_prob = jnp.exp(am_e.relmdl.logp(k2, test_s, sim_vals, sim_tr))
    normd_probs = sim_prob / mean_prob


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
    #plt.show()

    ### Define the SPM for inference ###
    mb = stratcona.SPMBuilder(mdl_name='E-Model')

    mb.add_hyperlatent('a_o_nom', dists.Normal, {'loc': 8.2, 'scale': 3.5}, transform=dists.transforms.SoftplusTransform())
    a_0_var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.1)])
    mb.add_hyperlatent('a_o_var', dists.Normal, {'loc': 13, 'scale': 3.8}, transform=a_0_var_tf)
    mb.add_latent('a_o', nom='a_o_nom', dev='a_o_var', chp=None, lot=None)

    mb.add_hyperlatent('e_aa_nom', dists.Normal, {'loc': 0.7, 'scale': 0.3})
    e_aa_var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.001)])
    mb.add_hyperlatent('e_aa_var', dists.Normal, {'loc': 12, 'scale': 5}, transform=e_aa_var_tf)
    mb.add_latent('e_aa', nom='e_aa_nom', dev='e_aa_var')

    mb.add_params(k=BOLTZ_EV, fit_var=0.1)

    mb.add_intermediate('ttf_base', e_model_ttf)
    mb.add_observed('ttf', dists.Normal, {'loc': 'ttf_base', 'scale': 'fit_var'}, 10)

    mb.add_fail_criterion('field_life', lambda ttf: ttf)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=6289383)
    am.set_field_use_conditions({'temp': 55 + CELSIUS_TO_KELVIN})

    # Sample some curves from the prior predictive
    test = stratcona.ReliabilityTest(
        {'e': {'lot': 1, 'chp': 1}},
        {'e': {'temp': 300, 'vg': 1.1}, 'em': {'temp': tm, 'vg': 1.1}, 'eh': {'temp': th, 'vg': 1.1}})
    rng = rand.key(48408)
    k1, k2 = rand.split(rng)
    prm_samples = am.relmdl.sample(k1, test, (100,), keep_sites=['a_o_nom', 'e_aa_nom'])
    pri_sample_probs = jnp.exp(am.relmdl.logp(k2, test, prm_samples, None, (100,)))
    pri_sample_probs = pri_sample_probs / (jnp.max(pri_sample_probs) * 2)
    spm_temps = jnp.full((100, 13,), jnp.linspace(300, 420, 13)).T
    pri_spm_vals = e_model_ttf(spm_temps, prm_samples['a_o_nom'], prm_samples['e_aa_nom'], BOLTZ_EV)
    pri_spm_vals = pri_spm_vals.T

    ### Inference the SPM ###
    am.set_test_definition(test_s)
    am.do_inference(ttfs['s']['e'])
    pprint(am.relmdl.hyl_beliefs)

    a_o_infd = am.relmdl.hyl_beliefs['a_o_nom']['loc']
    e_aa_infd = am.relmdl.hyl_beliefs['e_aa_nom']['loc']
    ttf_l = e_model_ttf(temp=tl, a_o=a_o_infd, e_aa=e_aa_infd, k=BOLTZ_EV)
    ttf_m = e_model_ttf(temp=tm, a_o=a_o_infd, e_aa=e_aa_infd, k=BOLTZ_EV)
    ttf_h = e_model_ttf(temp=th, a_o=a_o_infd, e_aa=e_aa_infd, k=BOLTZ_EV)

    mean_temps = jnp.linspace(300, 420, 13)
    mean_infd = e_model_ttf(temp=mean_temps, a_o=a_o_infd, e_aa=e_aa_infd, k=BOLTZ_EV)
    print(f'Inference model predicted MTTF: 85C={ttf_l}, 125C={ttf_m}, 145C={ttf_h}')

    # Sample curves from the posterior predictive
    rng = rand.key(27498)
    prm_samples = am.relmdl.sample(rng, test, (100,), keep_sites=['a_o_nom', 'e_aa_nom'])
    sample_probs = jnp.exp(am.relmdl.logp(k2, test, prm_samples, conditional=None, dims=(100,)))
    sample_probs = sample_probs / (jnp.max(sample_probs) * 2)
    spm_vals = e_model_ttf(spm_temps, prm_samples['a_o_nom'], prm_samples['e_aa_nom'], BOLTZ_EV)
    spm_vals, spm_temps = spm_vals.T, spm_temps.T

    temps_s = jnp.repeat(jnp.array([tl, tm, th]), 10)
    temps_l = jnp.repeat(jnp.array([tl, tm, th]), 10 * 7 * 7)

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
    #print(e_sampler.sample(rng_key, (10,)))

    #y_ttfs['s']['ie'] = funcs['ie'](x_temps, *fits['s']['ie']['e'][0])
    #y_ttfs['l']['ie'] = funcs['ie'](x_temps, *fits['l']['ie']['e'][0])
    y_ttfs['s']['p'] = funcs['p'](x_temps, *fits['s']['p']['e'][0])
    y_ttfs['l']['p'] = funcs['p'](x_temps, *fits['l']['p']['e'][0])

    sb.set_context('notebook')
    fig, p = plt.subplots(1, 1)
    p.grid()

    for i in range(100):
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

    plt.show()


if __name__ == '__main__':
    tddb_inference()
