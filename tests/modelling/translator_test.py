# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
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


def test_experiment_translator():
    expected = np.array([[2.3, 3, 40], [1.2, 6, 50]])
    map = gen_experiment_mapping(2, ['prm1', 'prm2', 'time'])
    tensor_form = translate_experiment({'exp1': {'prm1': 2.3, 'prm2': 3, 'time': 40},
                                        'exp2': {'prm1': 1.2, 'prm2': 6, 'time': 50}}, map, target='dontcare')
    internal_dict_form = translate_experiment(expected, map)
    readable_form = translate_experiment(expected, map, target='user')
    assert np.allclose(expected, tensor_form)
    assert np.allclose(internal_dict_form['prm2'], expected[:, 1])
    assert readable_form['exp1']['prm2'] == 3
    assert internal_dict_form['time'][1] == 50
    assert readable_form['exp2']['time'] == 50

    # Now test the optional named test functionality
    expected = np.array([[12.5, 340], [23, 120]])
    map = gen_experiment_mapping(['hard', 'soft'], ['temp', 'time'])
    tensor_form = translate_experiment({'soft': {'temp': 23, 'time': 120}, 'hard': {'time': 340, 'temp': 12.5}}, map)
    readable_form = translate_experiment(expected, map, target='user')
    assert np.allclose(expected, tensor_form)
    assert readable_form['hard']['time'] == 340
    assert readable_form['soft']['temp'] == 23


def test_observations_translator():
    expected = np.array([[[3.2], [3.4], [0], [0]], [[34], [42], [39], [41]]])
    readable = {'var2': [[34], [42], [39], [41]], 'var1': [[3.2], [3.4]]}
    map = gen_observation_mapping({'var1': 2, 'var2': 4}, 1)
    tensor_form = translate_observations(readable, map)
    assert np.allclose(tensor_form, expected)
    readable_form = translate_observations(expected, map, 'dontcare')
    assert np.allclose(readable_form['var2'], expected[1])
    assert np.allclose(readable_form['var1'], [[3.2], [3.4]])

    # Now test named experiment functionality
    expected = np.array([[[3.2, 6.3], [3.1, 7.1]]])
    readable = {'var1': {'yourexp': [6.3, 7.1], 'myexp': [3.2, 3.1]}}
    map = gen_observation_mapping({'var1': 2}, ['myexp', 'yourexp'])
    tensor_form = translate_observations(readable, map)
    assert np.allclose(tensor_form, expected)
    readable_form = translate_observations(expected, map, 'user')
    assert np.allclose(readable_form['var1']['myexp'], readable['var1']['myexp'])
    internal_form = translate_observations(expected, map, 'internal')
    assert np.allclose(internal_form['var1'], expected[0])
