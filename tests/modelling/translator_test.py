# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytensor as pt
import pymc

from stratcona.modelling.builder import ModelBuilder
from stratcona.modelling.tensor_dict_translator import *


def test_priors_translator():
    mb = ModelBuilder(mdl_name='Priors Translation')
    mb.add_latent_variable('a', pymc.HyperGeometric, {'N': 30, 'k': 5, 'n': 8})
    mb.add_latent_variable('b', pymc.Categorical, {'p': [0.2, 0.3, 0.3, 0.2]})
    mb.add_latent_variable('c', pymc.Categorical, {'p': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]})

    as_tensor = mb._prep_priors()
    as_dict = translate_priors(as_tensor, mb.priors_map)
    assert as_tensor[2, 0, 9] == 0.1
    assert as_tensor[0, 1, 0] == 5
    assert as_dict['b']['p'][2] == 0.3
    assert as_dict['a']['n'][0] == 8
