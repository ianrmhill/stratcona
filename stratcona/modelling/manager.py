# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

# I think it makes the most sense to have two model-level classes within Stratcona. The first is the model builder which
# needs to set everything up in order to spit out the needed PyMC model, but this requires a bunch of functionality and
# data which is no longer needed as soon as the model is built. To simplify the user experience, a model manager class
# here can be used to track the still-needed information and quickly call the various operations that can be performed
# on a model to simplify the user's experience in actually manipulating or computing quantities using the model.

from stratcona.assistants.probability import shorthand_compile # noqa: ImportNotAtTopOfFile
from stratcona.engine.inference import *
from stratcona.engine.metrics import *


class TestDesignManager:
    def __init__(self, model_builder):
        self.design_name = None
        self._test_model, self.latents_info, self.observed_info = model_builder.build_model()

        self._test_model = None

        self._handlers = {}
        self._compiled_funcs = {}

    def determine_best_test(self):
        # First compile any of the required graph computations that have not already been done
        if not 'ltnt_sampler' in self._compiled_funcs.keys():
            self._compiled_funcs['ltnt_sampler'] =\
                shorthand_compile('ltnt_sampler', self._test_model, self.latents_info, self.observed_info)
        if not 'ltnt_logp' in self._compiled_funcs.keys():
            self._compiled_funcs['ltnt_logp'] =\
                shorthand_compile('ltnt_logp', self._test_model, self.latents_info, self.observed_info)
        if not 'obs_logp' in self._compiled_funcs.keys():
            self._compiled_funcs['obs_logp'] =\
                shorthand_compile('obs_logp', self._test_model, self.latents_info, self.observed_info)

        best_test = None
        self._handlers['exp'].set_value(best_test)

    def infer_model(self, observations):
        self._handlers['obs'].set_value(observations)
        idata = inference_model(self._test_model, num_samples=3000)
        self._handlers['pri'].set_value(fit_latent_params_to_posterior_samples(self.latents_info, idata))

    def estimate_reliability(self):
        # TODO: Want to compile a lifespan sampler taking all the latents as inputs. This will require the lifespan to be
        #       a dependent variable within the test design model computation graph that already accounts for expected use
        #       conditions
        if not 'life_sampler' in self._compiled_funcs.keys():
            self._compiled_funcs['life_sampler'] =\
                shorthand_compile('life_sampler', self._test_model, self.latents_info, self.observed_info)
        bound = worst_case_quantile_credible_region(self._compiled_funcs['life_sampler'], 90)
