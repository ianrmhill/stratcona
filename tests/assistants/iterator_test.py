# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from stratcona.assistants.iterator import *


def test_discrete_iteration():
    my_vals = [0, 1, 4, 5]
    yielder = iter_sampler(my_vals, False)
    i = 0
    ended = False
    while not ended:
        try:
            assert yielder() == my_vals[i]
            i += 1
        except StopIteration:
            ended = True
    assert i == 4

    # Now test infinite cycling mode
    yielder = iter_sampler(my_vals)
    for i in range(10):
        assert yielder() == my_vals[i % 4]
        i += 1
