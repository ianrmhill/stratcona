# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from stratcona.engine.minimization import minimize, brent_minimize_std


def test_minimize():
    def test_parab(x, x_off):
        return (x + x_off) ** 2

    x_std, fx_std = brent_minimize_std(test_parab, {'x_off': 2}, (-50.0, 50.0))
    x_vec = minimize(test_parab, {'x_off': jnp.array([2, 3, -1])}, (-50.0, 50.0))

    assert round(x_std, 4) == -2
    assert jnp.all(jnp.round(x_vec, 5) == jnp.array([-2, -3, 1]))
