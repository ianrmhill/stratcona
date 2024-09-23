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
from scipy.stats import norm
from numpy.linalg import cholesky
from autograd import jacobian
import reliability
import pandas as pd

import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

import gracefall

import os
import sys
# This line adds the parent directory to the module search path so that the Stratcona module can be seen and imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stratcona

BOLTZ_EV = 8.617e-5
CELSIUS_TO_KELVIN = 273.15

SHOW_PLOTS = True


def weibull_inference():
    npyro.set_host_device_count(4)

    # Data manipulation helper function
    def convert(vals):
        return vals['ttf'].flatten()

    # Helper function to convert likelihoods to alpha values that result in clear plots
    def likelihood_to_alpha(probs, max_alpha=0.5):
        min_alpha = 0.1
        alphas = min_alpha + ((probs / jnp.max(probs)) * (max_alpha - min_alpha))
        return alphas

    # Weibull CDF
    def CDF(x, k, L):
        return 1 - jnp.exp(- (x / L) ** k)

    # Define the simulation test
    num_devs, num_chps, num_lots = 8, 5, 3
    th = 130 + CELSIUS_TO_KELVIN
    test_130 = stratcona.ReliabilityTest({'e': {'lot': num_lots, 'chp': num_chps}}, {'e': {'temp': th, 'vg': 1.1}})

    ######################################################
    # Define the SPM for Weibull analysis
    ######################################################
    mb_w = stratcona.SPMBuilder(mdl_name='weibull-2p')
    mb_w.add_hyperlatent('k_nom', dists.Normal, {'loc': 1.8, 'scale': 0.02}, transform=dists.transforms.SoftplusTransform())
    mb_w.add_hyperlatent('sc_nom', dists.Normal, {'loc': 1.9, 'scale': 0.02}, transform=dists.transforms.SoftplusTransform())
    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.01)])
    #mb_w.add_hyperlatent('k_dev', dists.Normal, {'loc': 2, 'scale': 0.2}, transform=var_tf)
    #mb_w.add_hyperlatent('sc_dev', dists.Normal, {'loc': 2, 'scale': 0.2}, transform=var_tf)
    mb_w.add_hyperlatent('k_lot', dists.Normal, {'loc': 10, 'scale': 0.2}, transform=var_tf)
    mb_w.add_hyperlatent('sc_lot', dists.Normal, {'loc': 10, 'scale': 0.2}, transform=var_tf)
    #mb_w.add_latent('k', nom='k_nom', dev='k_dev', chp=None, lot='k_lot')
    #mb_w.add_latent('sc', nom='sc_nom', dev='sc_dev', chp=None, lot='sc_lot')
    mb_w.add_latent('k', nom='k_nom', dev=None, chp=None, lot='k_lot')
    mb_w.add_latent('sc', nom='sc_nom', dev=None, chp=None, lot='sc_lot')
    mb_w.add_dependent('k_pos', lambda k: jnp.log(1 + jnp.exp(k)))
    mb_w.add_dependent('sc_pos', lambda sc: jnp.log(1 + jnp.exp(sc)))
    mb_w.add_measured('ttf', dists.Weibull, {'concentration': 'k_pos', 'scale': 'sc_pos'}, num_devs)

    am_w = stratcona.AnalysisManager(mb_w.build_model(), rng_seed=92633839)

    ######################################################
    # Simulate some Weibull distributed failure data
    ######################################################
    am_w.set_test_definition(test_130)
    ttfs = am_w.sim_test_measurements(rtrn_tr=True)

    sim_ks, sim_scs = ttfs['e_ttf_k_pos'], ttfs['e_ttf_sc_pos']
    ttfs = {'e': {'ttf': ttfs['e_ttf']}}
    ttfs_0 = {'e': {'ttf': ttfs['e']['ttf'][:, :, :, 0]}}
    ttfs_1 = {'e': {'ttf': ttfs['e']['ttf'][:, :, :, 1]}}
    ttfs_2 = {'e': {'ttf': ttfs['e']['ttf'][:, :, :, 2]}}

    fails_0, fails_1, fails_2 = convert(ttfs_0['e']), convert(ttfs_1['e']), convert(ttfs_2['e'])
    fails = jnp.concatenate((fails_0, fails_1, fails_2))
    fails_0, fails_1, fails_2 = np.sort(fails_0), np.sort(fails_1), np.sort(fails_2)
    n = len(fails_0)
    i = jnp.arange(1, n + 1)
    fail_order_s = (i - 0.5) / (n + 0.25)

    # Generate the function curves for the 'true' weibull distributions that were sampled
    sim_fits = []
    x = jnp.logspace(-2, 1, 50)
    for i in range(3):
        sim_fits.append(CDF(x, sim_ks[0, i], sim_scs[0, i]))

    # Sort the simulated data points for Weibull plotting and colour them according to lot
    n = len(fails)
    i = jnp.arange(1, n + 1)
    fail_order = (i - 0.5) / (n + 0.25)

    lots = jnp.array([0, 1, 2]).repeat(num_devs * num_chps)
    srtd_inds = jnp.argsort(fails)
    fails = fails[srtd_inds]
    srtd_lots = lots[srtd_inds]

    n0 = np.argwhere(srtd_lots != 0)
    f0 = jnp.delete(fails, n0)
    x0 = jnp.delete(fail_order, n0)
    n1 = np.argwhere(srtd_lots != 1)
    f1 = jnp.delete(fails, n1)
    x1 = jnp.delete(fail_order, n1)
    n2 = np.argwhere(srtd_lots != 2)
    f2 = jnp.delete(fails, n2)
    x2 = jnp.delete(fail_order, n2)


    ######################################################
    # Frequentist analysis of the failure data
    ######################################################
    fit_full = reliability.Fitters.Fit_Weibull_2P(fails.tolist(), show_probability_plot=False)
    fit_0 = reliability.Fitters.Fit_Weibull_2P(fails_0.tolist(), show_probability_plot=False)
    fit_1 = reliability.Fitters.Fit_Weibull_2P(fails_1.tolist(), show_probability_plot=False)
    fit_2 = reliability.Fitters.Fit_Weibull_2P(fails_2.tolist(), show_probability_plot=False)

    # Generate the function curves for the frequentist Weibull fits
    x = jnp.logspace(-2, 1, 50)
    fit_fails = CDF(x, fit_full.beta, fit_full.alpha)
    fit_fails_0 = CDF(x, fit_0.beta, fit_0.alpha)
    fit_fails_1 = CDF(x, fit_1.beta, fit_1.alpha)
    fit_fails_2 = CDF(x, fit_2.beta, fit_2.alpha)

    # Generate confidence interval (CI) function curves
    # y values are the quantiles to plot the time confidence intervals at
    # Functions to transform between linear and weibull scales
    fwdy = lambda y: jnp.log(jnp.fmax(1e-20, -jnp.log(jnp.fmax(1e-20, 1 - y))))
    bcky = lambda y: 1 - jnp.exp(-jnp.exp(y))
    fwdx = lambda x: jnp.log(jnp.fmax(1e-20, x))
    bckx = lambda x: jnp.exp(x)

    # Generate the set of y values to evaluate the CIs at
    quantile_min, quantile_max = 1e-3, 1 - 1e-3
    # Generate the array of quantiles in transform space then return to linear CDF quantiles
    arr = np.linspace(fwdy(quantile_min), fwdy(quantile_max), 50)
    ci_q = bcky(arr)
    ci_q_rev = 1 - ci_q

    ci = 0.95
    # Convert CI to Z, ppf is inverse of CDF, in this case of a standard normal distribution
    z = -norm.ppf((1 - ci) / 2)

    # This is the Weibull survival function which is just 1 - CDF, then rearranged for t
    def v(q, alpha, beta):  # v = ln(t)
        return (1 / beta) * jnp.log(-jnp.log(q)) + jnp.log(alpha)

    dv_da = jax.jacfwd(v, 1)
    dv_db = jax.jacfwd(v, 2)

    def var_v(a, a_se, b, b_se, cov_ab, y):
        return (((dv_da(y, a, b) ** 2) * (a_se ** 2)) +
                ((dv_db(y, a, b) ** 2) * (b_se ** 2)) +
                (2 * dv_da(y, a, b) * dv_db(y, a, b) * cov_ab))

    alpha, alpha_se, beta, beta_se = fit_full.alpha, fit_full.alpha_SE, fit_full.beta, fit_full.beta_SE
    ab_cov, gamma = fit_full.Cov_alpha_beta, fit_full.gamma

    # Now need to evaluate the CI at each quantile value
    v_lower = v(ci_q_rev, alpha, beta) - z * (var_v(alpha, alpha_se, beta, beta_se, ab_cov, ci_q_rev) ** 0.5)
    v_upper = v(ci_q_rev, alpha, beta) + z * (var_v(alpha, alpha_se, beta, beta_se, ab_cov, ci_q_rev) ** 0.5)
    x_lower = jnp.exp(v_lower) + gamma  # transform back from ln(t)
    x_upper = jnp.exp(v_upper) + gamma

    ######################################################
    # Plot the simulated data and frequentist fits to that data
    ######################################################
    sb.set_context('notebook')
    fig, p = plt.subplots(1, 1)
    p.grid()

    #p.plot(fails_0, fail_order_s, color='orchid', linestyle='', marker='.', markersize=8)
    #p.plot(fails_1, fail_order_s, color='darkorchid', linestyle='', marker='.', markersize=8)
    #p.plot(fails_2, fail_order_s, color='mediumvioletred', linestyle='', marker='.', markersize=8)

    p.plot(x, fit_fails, color='darkblue', linewidth=2)
    #p.plot(x, fit_fails_0, color='orchid', linestyle='-', linewidth=2)
    #p.plot(x, fit_fails_1, color='darkorchid', linestyle='-', linewidth=2)
    #p.plot(x, fit_fails_2, color='mediumvioletred', linestyle='-', linewidth=2)

    # Plot the confidence intervals for the frequentist fits
    xstack = np.hstack([x_lower, x_upper[::-1]])
    ystack = np.hstack([ci_q, ci_q[::-1]])
    polygon = np.column_stack([xstack, ystack])
    col = PolyCollection([polygon], color='darkblue', alpha=0.3)
    p.add_collection(col, autolim=False)

    p.plot(f0, x0, color='darkorange', linestyle='', marker='.', markersize=8)
    p.plot(f1, x1, color='sienna', linestyle='', marker='.', markersize=8)
    p.plot(f2, x2, color='gold', linestyle='', marker='.', markersize=8)

    p.plot(x, sim_fits[0], color='darkorange', linestyle='--', linewidth=2)
    p.plot(x, sim_fits[1], color='sienna', linestyle='--', linewidth=2)
    p.plot(x, sim_fits[2], color='gold', linestyle='--', linewidth=2)



    # Set up the plot axes scales and bounds
    p.set_xscale('function', functions=(fwdx, bckx))
    ln_min, ln_max = jnp.log(min(fails)), jnp.log(max(fails))
    lim_l = jnp.exp(ln_min - (0.01 * (ln_max - ln_min)))
    lim_h = jnp.exp(ln_max + (0.05 * (ln_max - ln_min)))
    p.set_xlim(lim_l, lim_h)
    p.set_yscale('function', functions=(fwdy, bcky))
    p.set_ylim(0.01, 0.99)
    weibull_ticks = [0.01, 0.02, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.96, 0.99]
    tick_labels = ['1%', '2%', '5%', '10%', '25%', '50%', '75%', '90%', '96%', '99%']
    p.set_yticks(weibull_ticks, tick_labels)

    p.set_xlabel('Time to Failure (years)')
    p.set_ylabel('CDF [ln(ln(1-F))]')
    #plt.show()


    ######################################################
    # Now inference the SPM on the simulated data
    ######################################################
    test_130_sing = stratcona.ReliabilityTest({'e': {'lot': 1, 'chp': 1, 'ttf': 1}}, {'e': {'temp': th, 'vg': 1.1}})
    k1, k2, k3, k4 = rand.split(rand.key(428027234), 4)
    eval_sites = ['e_ttf_k_dev', 'e_ttf_sc_dev', 'e_k_lot', 'e_sc_lot',
                  'k_nom', 'sc_nom', 'k_dev', 'sc_dev', 'k_lot', 'sc_lot']

    # Priors for the SPM are defined here
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

    ######################################################
    # Generate the second plot that shows the Bayesian inference approach
    ######################################################
    sb.set_context('notebook')
    fig, p = plt.subplots(1, 1)
    p.grid()
    # Plot the probabilistic fits
    for i in range(len(pri_fits)):
        p.plot(x, pri_fits[i].flatten(), alpha=float(pri_probs[i]), color='skyblue')
    for i in range(len(pst_fits)):
        p.plot(x, pst_fits[i].flatten(), alpha=float(pst_probs[i]), color='darkblue')

    #p.plot(x, fit_fails, color='indigo', linewidth=2)
    #p.plot(x, fit_fails_0, color='orchid', linestyle='--', linewidth=2)
    #p.plot(x, fit_fails_1, color='darkorchid', linestyle='--', linewidth=2)
    #p.plot(x, fit_fails_2, color='mediumvioletred', linestyle='--', linewidth=2)

    #p.plot(fails_0, fail_order_s, color='orchid', linestyle='', marker='.', markersize=8)
    #p.plot(fails_1, fail_order_s, color='darkorchid', linestyle='', marker='.', markersize=8)
    #p.plot(fails_2, fail_order_s, color='mediumvioletred', linestyle='', marker='.', markersize=8)

    p.plot(x, sim_fits[0], color='darkorange', linestyle='--', linewidth=2)
    p.plot(x, sim_fits[1], color='sienna', linestyle='--', linewidth=2)
    p.plot(x, sim_fits[2], color='gold', linestyle='--', linewidth=2)

    p.plot(f0, x0, color='darkorange', linestyle='', marker='.', markersize=8)
    p.plot(f1, x1, color='sienna', linestyle='', marker='.', markersize=8)
    p.plot(f2, x2, color='gold', linestyle='', marker='.', markersize=8)

    p.set_xscale('function', functions=(fwdx, bckx))
    ln_min, ln_max = jnp.log(min(fails)), jnp.log(max(fails))
    lim_l = jnp.exp(ln_min - (0.01 * (ln_max - ln_min)))
    lim_h = jnp.exp(ln_max + (0.05 * (ln_max - ln_min)))
    p.set_xlim(lim_l, lim_h)
    p.set_yscale('function', functions=(fwdy, bcky))
    p.set_ylim(0.01, 0.99)
    weibull_ticks = [0.01, 0.02, 0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.96, 0.99]
    tick_labels = ['1%', '2%', '5%', '10%', '25%', '50%', '75%', '90%', '96%', '99%']
    p.set_yticks(weibull_ticks, tick_labels)

    p.set_xlabel('Time to Failure (years)')
    p.set_ylabel('CDF [ln(ln(1-F))]')
    plt.show()


if __name__ == '__main__':
    weibull_inference()
