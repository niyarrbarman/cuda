from math import ceil

import cuda.tile as ct
import cupy as cp
import numpy as np


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)

    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    result = a_tile + b_tile

    ct.store(c, index=(pid,), tile=result)


def test():
    vec_size = 2**12
    tile_size = 2**4

    a = cp.random.uniform(-1, 1, vec_size)
    b = cp.random.uniform(-1, 1, vec_size)
    c = cp.zeros_like(a)

    grid_size = (ceil(vec_size / tile_size), 1, 1)

    ct.launch(cp.cuda.get_current_stream(), grid_size, vector_add, (a, b, c, tile_size))

    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)

    excepted = a_np + b_np
    actual = cp.asnumpy(c)

    print(np.testing.assert_allclose(excepted, actual))


if __name__ == "__main__":
    test()
