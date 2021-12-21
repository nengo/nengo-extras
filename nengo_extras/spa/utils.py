from __future__ import absolute_import

import numpy as np


def circconv(a, b, k=1, invert_a=False, invert_b=False, axis=-1):
    """A reference Numpy implementation of circular convolution.

    If ``k == 1``, returns ``a * b``, where ``*`` represents circular
    convolution.

    If ``k > 1``, return ``a * b * ... * b``, with ``k`` copies of ``b``.

    Parameters
    ----------
    a, b : array_like
        Arrays to convolve.
    k : int (optional)
        Number of times to convolve ``b`` with ``a``. Defaults to 1.
    invert_a : boolean
        Whether to invert ``a``.
    invert_b : boolean
        Whether to invert ``b``.
    axis : int (optional)
        Axis along which to perform the convolution. Defaults to the last axis.
    """
    A = np.fft.fft(a, axis=axis)
    B = np.fft.fft(b, axis=axis)
    if invert_a:
        A = A.conj()
    if invert_b:
        B = B.conj()
    if k != 1:
        B = B ** k
    return np.fft.ifft(A * B, axis=axis).real


def _shared_factor_set(k):
    """Returns a set of all numbers sharing a common factor with `k`"""
    f = []
    sqrt = lambda x: int(np.sqrt(x))
    for i in range(2, sqrt(k) + 1):
        if k % i == 0:
            ki = int(k / i)
            f.extend(p * i for p in range(1, ki))
            f.extend(p * ki for p in range(1, i))

    return set(f)


def cyclic_vector(d, k, n=None, rng=np.random):
    """Leaves a target vector unchanged after k convolutions.

    For example, if a is any vector and b = cyclic_vector(d, 4):

        a * b * b * b * b = a

    where * is the circular convolution operator. By corollary:

        b * b * b * b = [1, 0, ..., 0]

    Parameters
    ----------
    d : int
        The number of dimensions of the generated vector(s).
    k : int
        The number of convolutions required to reproduce the input.
    n : int (optional)
        The number of vectors to generate.
    rng : random number generator (optional)
        The random number generator to use.

    Output
    ------
    u : (n, d) array
        Array of vector(s). If 'n' is None, the shape will be `(d,)`.
    """
    d, k = int(d), int(k)
    if k < 2:
        raise ValueError("'k' must be at least 2 (got %d)" % k)
    if d < 3:
        raise ValueError("'d' must be at least 3 (got %d)" % d)

    d2 = (d - 1) // 2
    nn = 1 if n is None else int(n)

    # Pick roots r such that r**k == 1
    roots = np.exp(2.0j * np.pi / k * np.arange(k))
    rootpow = rng.randint(0, k, size=(nn, d2))

    # Ensure at least one root power in each vector is coprime with k, so that
    # no vector will take LESS than k convolutions to reproduce itself.
    coprimes = set(range(1, k)) - _shared_factor_set(k)
    for i in range(nn):
        # TODO: better method than rejection sampling?
        while not any(p in coprimes for p in rootpow[i]):
            rootpow[i] = rng.randint(0, k, size=d2)

    # Create array of Fourier coefficients such that U**k == ones(d)
    U = np.ones((nn, d), dtype=np.complex128)
    U[:, 1 : 1 + d2] = roots[rootpow]

    # For even k, U[:, d2+1] = 1 or -1 are both valid
    if k % 2 == 0:
        U[:, 1 + d2] = 2 * rng.randint(0, 2, size=nn) - 1

    # Respect Fourier symmetry conditions for real vectors
    U[:, -d2:] = U[:, d2:0:-1].conj()

    u = np.fft.ifft(U, axis=1).real
    return u[0] if n is None else u
