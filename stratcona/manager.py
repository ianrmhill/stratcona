# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from functools import partial
import jax.random as rand

from stratcona import engine
from stratcona.engine.inference import inference_model, inf_is
from stratcona.engine.bed import pred_bed
from stratcona.modelling.relmodel import ReliabilityModel, ReliabilityRequirement, TestDef


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

    def sim_test_meas(self, num=(), rtrn_tr=False):
        rng = self._derive_key()
        if rtrn_tr:
            return self.relmdl.sample(rng, self.test.dims, self.test.conds, num)
        else:
            return self.relmdl.sample(rng, self.test.dims, self.test.conds, num, self.relmdl.observes)

    def do_inference(self, observations, test: TestDef = None, auto_update_prior=True):
        rng = self._derive_key()
        test_info = test if test is not None else self.test
        inf_mdl = partial(self.relmdl.spm, test_info.dims, test_info.conds, self.relmdl.hyl_beliefs, self.relmdl.param_vals)
        new_prior = inference_model(inf_mdl, self.relmdl.hyl_info, observations, rng)
        if auto_update_prior:
            self.relmdl.hyl_beliefs = new_prior

    def do_inference_is(self, observations, test: TestDef = None, n_x=10_000, n_v=500):
        rng = self._derive_key()
        test_info = test if test is not None else self.test
        new_prior, perf_stats = inf_is(rng, self.relmdl, test_info, observations, self.relmdl.obs_noise, n_x, n_v)
        self.relmdl.hyl_beliefs = new_prior
        return perf_stats

    def evaluate_reliability(self, predictor, num_samples=300_000):
        rng = self._derive_key()
        pred_site = f'field_{predictor}'
        samples = self.relmdl.sample(rng, self.field_test.dims, self.field_test.conds, (num_samples,),
                                     keep_sites=(pred_site,), compute_predictors=True)

        lifespan = self.relreq.type(samples[pred_site], self.relreq.quantile)

        if type(lifespan) != list:
            if lifespan >= self.relreq.target_lifespan:
                print(f'Target lifespan of {self.relreq.target_lifespan} met! Predicted: {lifespan}.')
            else:
                print(f'Target lifespan of {self.relreq.target_lifespan} not met! Predicted: {lifespan}.')

        return lifespan

    def determine_best_test(self, n_d, n_y, n_v, n_x, exp_sampler, u_funcs=engine.bed.eig):
        rng = self._derive_key()
        return pred_bed(rng, exp_sampler, n_d, n_y, n_v, n_x, self.relmdl, u_funcs, self.field_test)
