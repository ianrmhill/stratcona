# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpyro as npyro
import numpyro.distributions as dists
# Device count has to be set before importing jax
npyro.set_host_device_count(4)

import jax
import jax.numpy as jnp
import jax.random as rand
from jax.scipy.stats import norm

import reliability
import time

import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.lines as pltlines

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


# Helper function to convert likelihoods to alpha values that result in clear plots
def likelihood_to_alpha(probs, max_alpha=0.5):
    min_alpha = 0.1
    alphas = min_alpha + ((probs / jnp.max(probs)) * (max_alpha - min_alpha))
    return alphas


# Weibull weibull_cdf
def weibull_cdf(x, k, L):
    return 1 - jnp.exp(- (x / L) ** k)


def weibull_inference():
    """
    This procedure recreates the Weibull failure analysis case study from the IRPS 2025 paper.
    """

    ######################################################
    # Simulated data generation and model construction
    ######################################################

    # Define the simulation test
    num_devs, num_chps, num_lots = 8, 5, 3
    test_conds = {'e': {'temp': 130 + CELSIUS_TO_KELVIN, 'vg': 1.1}}
    test_130 = stratcona.ReliabilityTest({'e': {'lot': num_lots, 'chp': num_chps}}, test_conds)
    test_130_single = stratcona.ReliabilityTest({'e': {'lot': 1, 'chp': 1, 'ttf': 1}}, test_conds)

    # Define the SPM for simulation and Weibull analysis
    mb_w = stratcona.SPMBuilder(mdl_name='weibull-2p')
    mb_w.add_hyperlatent('k_nom', dists.Normal, {'loc': 1.8, 'scale': 0.02}, transform=dists.transforms.SoftplusTransform())
    mb_w.add_hyperlatent('sc_nom', dists.Normal, {'loc': 1.9, 'scale': 0.02}, transform=dists.transforms.SoftplusTransform())
    var_tf = dists.transforms.ComposeTransform([dists.transforms.SoftplusTransform(), dists.transforms.AffineTransform(0, 0.01)])
    mb_w.add_hyperlatent('k_lot', dists.Normal, {'loc': 10, 'scale': 0.2}, transform=var_tf)
    mb_w.add_hyperlatent('sc_lot', dists.Normal, {'loc': 10, 'scale': 0.2}, transform=var_tf)
    mb_w.add_latent('k', nom='k_nom', dev=None, chp=None, lot='k_lot')
    mb_w.add_latent('sc', nom='sc_nom', dev=None, chp=None, lot='sc_lot')
    mb_w.add_intermediate('k_pos', lambda k: jnp.log(1 + jnp.exp(k)))
    mb_w.add_intermediate('sc_pos', lambda sc: jnp.log(1 + jnp.exp(sc)))
    mb_w.add_observed('ttf', dists.Weibull, {'concentration': 'k_pos', 'scale': 'sc_pos'}, num_devs)

    am_w = stratcona.AnalysisManager(mb_w.build_model(), rng_seed=92633839)

    # Simulate some Weibull distributed failure data
    am_w.set_test_definition(test_130)
    ttfs = am_w.sim_test_measurements()
    lot_vals = (ttfs['e_k_lot'], ttfs['e_sc_lot'])
    sim_ks, sim_scs = ttfs['e_ttf_k_pos'], ttfs['e_ttf_sc_pos']

    # Extract and order the simulated failure times for plotting
    ttfs = {'e': {'ttf': ttfs['e_ttf']}}
    ttfs_0 = {'e': {'ttf': ttfs['e']['ttf'][:, :, 0]}}
    ttfs_1 = {'e': {'ttf': ttfs['e']['ttf'][:, :, 1]}}
    ttfs_2 = {'e': {'ttf': ttfs['e']['ttf'][:, :, 2]}}
    fails_0, fails_1, fails_2 = convert(ttfs_0['e']), convert(ttfs_1['e']), convert(ttfs_2['e'])
    fails = jnp.concatenate((fails_0, fails_1, fails_2))
    fails_0, fails_1, fails_2 = jnp.sort(fails_0), jnp.sort(fails_1), jnp.sort(fails_2)

    # Extract the function curves for the 'true' weibull distributions that were sampled
    sim_dists = []
    x = jnp.logspace(-2, 1, 50)
    for i in range(3):
        sim_dists.append(weibull_cdf(x, sim_ks[i], sim_scs[i]))

    # Sort the simulated data points by failure time for Weibull plotting and colour them according to lot
    n = len(fails)
    i = jnp.arange(1, n + 1)
    fail_order = (i - 0.5) / (n + 0.25)
    lots = jnp.array([0, 1, 2]).repeat(num_devs * num_chps)
    srtd_inds = jnp.argsort(fails)
    fails = fails[srtd_inds]
    srtd_lots = lots[srtd_inds]
    # Extract points from each lot from the full sorted data so they can be plotted by colour later
    n0 = jnp.argwhere(srtd_lots != 0)
    f0 = jnp.delete(fails, n0)
    x0 = jnp.delete(fail_order, n0)
    n1 = jnp.argwhere(srtd_lots != 1)
    f1 = jnp.delete(fails, n1)
    x1 = jnp.delete(fail_order, n1)
    n2 = jnp.argwhere(srtd_lots != 2)
    f2 = jnp.delete(fails, n2)
    x2 = jnp.delete(fail_order, n2)

    ######################################################
    # Frequentist analysis of the failure data
    ######################################################
    fit_full = reliability.Fitters.Fit_Weibull_2P(fails.tolist(), show_probability_plot=False)
    fit_0 = reliability.Fitters.Fit_Weibull_2P(fails_0.tolist(), show_probability_plot=False)
    fit_1 = reliability.Fitters.Fit_Weibull_2P(fails_1.tolist(), show_probability_plot=False)
    fit_2 = reliability.Fitters.Fit_Weibull_2P(fails_2.tolist(), show_probability_plot=False)

    # Generate the function curves for the MLE fits
    x = jnp.logspace(-2, 1, 50)
    fit_fails = weibull_cdf(x, fit_full.beta, fit_full.alpha)
    # Individual lot fits, currently unused
    fit_fails_0 = weibull_cdf(x, fit_0.beta, fit_0.alpha)
    fit_fails_1 = weibull_cdf(x, fit_1.beta, fit_1.alpha)
    fit_fails_2 = weibull_cdf(x, fit_2.beta, fit_2.alpha)

    # Remainder of this section is used to generate confidence interval (CI) function curves
    # y values are the quantiles to plot the time confidence intervals at

    # Need these functions to transform between linear and weibull axis scales
    fwdy = lambda y: jnp.log(jnp.fmax(1e-20, -jnp.log(jnp.fmax(1e-20, 1 - y))))
    bcky = lambda y: 1 - jnp.exp(-jnp.exp(y))
    fwdx = lambda x: jnp.log(jnp.fmax(1e-20, x))
    bckx = lambda x: jnp.exp(x)

    # Generate the set of y values to evaluate the CIs at
    quantile_min, quantile_max = 1e-3, 1 - 1e-3
    # Generate the array of quantiles in transform space then return to linear weibull_cdf quantiles
    arr = jnp.linspace(fwdy(quantile_min), fwdy(quantile_max), 50)
    ci_q = bcky(arr)
    ci_q_rev = 1 - ci_q

    ci = 0.95
    # Convert CI to Z, ppf is inverse of weibull_cdf, in this case of a standard normal distribution
    z = -norm.ppf((1 - ci) / 2)

    # This is the Weibull survival function which is just 1 - weibull_cdf, then rearranged for t
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
    # Plot the simulated data and MLE fit to that data
    ######################################################
    sb.set_context('notebook')
    sb.set_theme(style='ticks', font='Times New Roman')
    fig, p = plt.subplots(1, 1)
    p.grid()
    # Plotting order below matters to get a visually sensible layering (i.e., confidence interval should be background)

    # Plot the confidence intervals for the frequentist fits
    xstack = jnp.hstack([x_lower, x_upper[::-1]])
    ystack = jnp.hstack([ci_q, ci_q[::-1]])
    polygon = jnp.column_stack([xstack, ystack])
    col = PolyCollection([polygon], color='darkblue', alpha=0.3)
    p.add_collection(col, autolim=False)

    # Plot the underlying Weibull distributions for the three lots
    p.plot(x, sim_dists[0], color='darkorange', linestyle='--', linewidth=2, label=f'Lot 1')
    p.plot(x, sim_dists[1], color='sienna', linestyle='--', linewidth=2, label=f'Lot 2')
    p.plot(x, sim_dists[2], color='gold', linestyle='--', linewidth=2, label=f'Lot 3')

    # Plot the fitted curve
    p.plot(x, fit_fails, color='darkblue', linewidth=2)

    # Plot the failure times, sorted jointly but coloured according to lot
    p.plot(f0, x0, color='darkorange', linestyle='', marker='.', markersize=6)
    p.plot(f1, x1, color='sienna', linestyle='', marker='.', markersize=6)
    p.plot(f2, x2, color='gold', linestyle='', marker='.', markersize=6)

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
    p.set_ylabel('weibull_cdf [ln(ln(1-F))]')

    hndl, lbls = p.get_legend_handles_labels()
    lgnd1 = pltlines.Line2D([0], [0], color='darkblue', linestyle='-')
    lgnd2 = pltlines.Line2D([0], [0], color='black', linestyle='', marker='.')
    hndl.insert(0, lgnd1)
    lbls.insert(0, 'MLE fit to data')
    hndl.insert(0, lgnd2)
    lbls.insert(0, 'Simulated failure data')
    p.legend(hndl, lbls, loc='lower right')

    # Add annotations that highlight key elements for readers
    p.annotate('Lot 1 distribution outside 95%\nconfidence interval of MLE fit', (3.5, 0.96), (0.3, 0.85),
               arrowprops={'arrowstyle': 'simple', 'color': 'black'})

    ######################################################
    # Bayesian inference of the SPM on the simulated data
    ######################################################
    k1, k2, k3, k4 = rand.split(rand.key(428027234), 4)
    eval_sites = ['e_ttf_k_dev', 'e_ttf_sc_dev', 'e_k_lot', 'e_sc_lot',
                  'k_nom', 'sc_nom', 'k_dev', 'sc_dev', 'k_lot', 'sc_lot']

    # Priors for the SPM are defined here
    am_w.relmdl.hyl_beliefs = {'k_nom': {'loc': 2.0, 'scale': 1.5},
                               'k_lot': {'loc': 15, 'scale': 5},
                               'sc_nom': {'loc': 1.8, 'scale': 0.8},
                               'sc_lot': {'loc': 15, 'scale': 5}}

    # Generate prior samples for plotting the prior predictive distribution
    prm_samples = am_w.relmdl.sample(k1, test_130_single, (400,))
    ltnt_vals = {site: data for site, data in prm_samples.items() if site in eval_sites}
    pri_probs = jnp.exp(am_w.relmdl.logp(k2, test_130_single, ltnt_vals, prm_samples, dims=(400,)))
    pri_probs = likelihood_to_alpha(pri_probs, 0.4).flatten()

    x = jnp.logspace(-2, 1, 50)
    pri_fits = weibull_cdf(x, prm_samples['e_ttf_k_pos'], prm_samples['e_ttf_sc_pos'])

    start_time = time.time()
    am_w.do_inference(ttfs, test_130)
    print(f'Inference time taken: {time.time() - start_time}')
    print(am_w.relmdl.hyl_beliefs)
    post_cond = {'e': {'k_nom': am_w.relmdl.hyl_beliefs['k_nom']['loc'], 'sc_nom': am_w.relmdl.hyl_beliefs['sc_nom']['loc'],
                       'k_lot': am_w.relmdl.hyl_beliefs['k_lot']['loc'], 'sc_lot': am_w.relmdl.hyl_beliefs['sc_lot']['loc']}}

    prm_samples = am_w.relmdl.sample(k3, test_130_single, (400,))
    ltnt_vals = {site: data for site, data in prm_samples.items() if site in eval_sites}
    pst_probs = jnp.exp(am_w.relmdl.logp(k4, test_130_single, ltnt_vals, prm_samples, dims=(400,)))
    pst_probs = likelihood_to_alpha(pst_probs, 0.4).flatten()

    pst_fits = weibull_cdf(x, prm_samples['e_ttf_k_pos'], prm_samples['e_ttf_sc_pos'])

    # Determine the probability of the simulation traces under the posterior inference model
    k1, k2 = rand.split(rand.key(7932854))
    mean_prob = jnp.exp(am_w.relmdl.logp(k1, test_130_single, {'e_k_lot': jnp.array([0.0]),
                                                             'e_sc_lot': jnp.array([0.0])}, post_cond))
    p1 = jnp.exp(am_w.relmdl.logp(k1, test_130_single, {'e_k_lot': lot_vals[0][0], 'e_sc_lot': lot_vals[1][0]}, post_cond))
    p2 = jnp.exp(am_w.relmdl.logp(k1, test_130_single, {'e_k_lot': lot_vals[0][1], 'e_sc_lot': lot_vals[1][1]}, post_cond))
    p3 = jnp.exp(am_w.relmdl.logp(k1, test_130_single, {'e_k_lot': lot_vals[0][2], 'e_sc_lot': lot_vals[1][2]}, post_cond))
    pst_sim_probs = jnp.array([p1, p2, p3]) / mean_prob
    print(f'Posterior normalized likelihood of simulation lot variability: {pst_sim_probs}')

    ######################################################
    # Render the SPM for viewing
    ######################################################
    dims = test_130.config
    conds = test_130.conditions
    priors = am_w.relmdl.hyl_beliefs
    params = am_w.relmdl.param_vals
    npyro.render_model(am_w.relmdl.spm, model_args=(dims, conds, priors, params),
                       filename='../renders/weibull_model.png')

    ######################################################
    # Generate the second plot that shows the Bayesian inference approach
    ######################################################
    fig, p = plt.subplots(1, 1)
    p.grid()
    # Plot the probabilistic fits
    for i in range(len(pri_fits)):
        lbl = 'Prior predictive distribution' if i == 0 else None
        p.plot(x, pri_fits[i].flatten(), alpha=float(pri_probs[i]), color='skyblue', label=lbl)
    for i in range(len(pst_fits)):
        lbl = 'Posterior predictive distribution' if i == 0 else None
        p.plot(x, pst_fits[i].flatten(), alpha=float(pst_probs[i]), color='darkblue', label=lbl)

    p.plot(x, sim_dists[0], color='darkorange', linestyle='--', linewidth=2)
    p.plot(x, sim_dists[1], color='sienna', linestyle='--', linewidth=2)
    p.plot(x, sim_dists[2], color='gold', linestyle='--', linewidth=2)

    p.plot(f0, x0, color='darkorange', linestyle='', marker='.', markersize=6)
    p.plot(f1, x1, color='sienna', linestyle='', marker='.', markersize=6)
    p.plot(f2, x2, color='gold', linestyle='', marker='.', markersize=6)

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

    hndl, lbls = p.get_legend_handles_labels()
    lgnd1 = pltlines.Line2D([0], [0], color='black', linestyle='--')
    lgnd2 = pltlines.Line2D([0], [0], color='black', linestyle='', marker='.')
    hndl.insert(0, lgnd1)
    lbls.insert(0, 'Lot simulation distributions')
    hndl.insert(0, lgnd2)
    lbls.insert(0, 'Simulated data, coloured by lot')

    # Add the custom legend
    leg = p.legend(hndl, lbls, loc='lower right')
    for lbl in leg.legend_handles:
        lbl.set_alpha(1)
    # Add annotation to highlight that Lot 1 is comfortably within Bayesian posterior predictive
    p.annotate(f'Lot 1 posterior normalized\nlikelihood: {round(float(pst_sim_probs[0] * 100), 1)}%',
               (3.5, 0.96), (0.4, 0.85),
               arrowprops={'arrowstyle': 'simple', 'color': 'black'})
    plt.show()


if __name__ == '__main__':
    weibull_inference()
