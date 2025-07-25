# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from functools import partial
from matplotlib import pyplot as plt

import jax.random as rand

from stratcona import engine
from stratcona.engine.inference import inference_model, custom_mhgibbs_new, inf_is_new
from stratcona.engine.bed import pred_bed_apr25, eig
from stratcona.modelling.relmodel import ReliabilityModel, ReliabilityTest, ReliabilityRequirement, TestDef


class AnalysisManager:
    def __init__(self, model: ReliabilityModel, rng_seed: int, rel_req: ReliabilityRequirement = None):
        self.relmdl = model
        self.relreq = rel_req
        self.test = None
        self.field_test = None
        self.rng_key = rand.key(rng_seed)

    def _derive_key(self):
        self.rng_key, for_use = rand.split(self.rng_key)
        return for_use

    def set_test_definition(self, test_def: TestDef):
        self.test = test_def

    def set_field_use_conditions(self, conds):
        self.field_test = TestDef('fielduse', {'field': {'lot': 1, 'chp': 1}}, {'field': conds})

    def update_priors(self, new_priors):
        self.relmdl.hyl_beliefs = new_priors

    def sim_test_measurements(self, alt_priors=None, rtrn_tr=False, num=()):
        rng = self._derive_key()
        return self.relmdl.sample(rng, self.test, num_samples=num, alt_priors=alt_priors, full_trace=rtrn_tr)

    def sim_test_meas_new(self, num=()):
        rng = self._derive_key()
        return self.relmdl.sample_new(rng, self.test.dims, self.test.conds, num, self.relmdl.observes)

    def do_inference(self, observations, test: ReliabilityTest = None, auto_update_prior=True):
        rng = self._derive_key()
        test_info = test if test is not None else self.test
        inf_mdl = partial(self.relmdl.spm, test_info.dims, test_info.conds, self.relmdl.hyl_beliefs, self.relmdl.param_vals)
        new_prior = inference_model(inf_mdl, self.relmdl.hyl_info, observations, rng)
        if auto_update_prior:
            self.relmdl.hyl_beliefs = new_prior

    def do_inference_mhgibbs(self, observations, test: ReliabilityTest = None, num_chains=10, n_v=100, beta=0.5):
        rng = self._derive_key()
        test_info = test if test is not None else self.test
        new_prior, perf_stats = custom_mhgibbs_new(rng, self.relmdl, test_info, observations, self.relmdl.obs_noise, num_chains, n_v, beta)
        self.relmdl.hyl_beliefs = new_prior
        print(perf_stats)

    def do_inference_is(self, observations, test: TestDef = None, n_x=10_000, n_v=500):
        rng = self._derive_key()
        test_info = test if test is not None else self.test
        new_prior, perf_stats = inf_is_new(rng, self.relmdl, test_info, observations, self.relmdl.obs_noise, n_x, n_v)
        self.relmdl.hyl_beliefs = new_prior
        return perf_stats

    def evaluate_reliability(self, predictor, num_samples=300_000, plot_results=False):
        rng = self._derive_key()
        pred_site = f'field_{predictor}'
        samples = self.relmdl.sample_new(rng, self.field_test.dims, self.field_test.conds, (num_samples,),
                                         keep_sites=(pred_site,), compute_predictors=True)

        lifespan = self.relreq.type(samples[pred_site], self.relreq.quantile)

        if type(lifespan) != list:
            if lifespan >= self.relreq.target_lifespan:
                print(f'Target lifespan of {self.relreq.target_lifespan} met! Predicted: {lifespan}.')
            else:
                print(f'Target lifespan of {self.relreq.target_lifespan} not met! Predicted: {lifespan}.')

        if plot_results:
            # TODO: May be better to plot the CDF as opposed to the PDF for this visualization
            fig, p = plt.subplots(1, 1)
            p.hist(samples[pred_site], 300, density=True, color='grey', histtype='stepfilled')
            p.axvline(float(lifespan), 0, 1, color='orange', linestyle='solid',
                      label=f'Q{self.relreq.quantile}%-LBCI: {round(float(lifespan), 2)}')
            p.axvline(float(self.relreq.target_lifespan), 0, 1, color='orange', linestyle='dashed',
                      label=f'Target lifespan: {round(float(self.relreq.target_lifespan), 2)}')

            p.legend()
            p.set_xlabel('Failure Time (years)', fontsize='medium')
            p.set_ylabel('Probability Density')

        return lifespan

    def determine_best_test_apr25(self, n_d, n_y, n_v, n_x, exp_sampler, u_funcs=engine.bed.eig):
        rng = self._derive_key()
        return pred_bed_apr25(rng, exp_sampler, n_d, n_y, n_v, n_x, self.relmdl,
                              u_funcs, self.field_test)
