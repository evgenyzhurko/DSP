import numpy as np


def _w(n, l):
    return np.exp(-1j * 2 * np.pi * n / l)


def fft(x):
    size = len(x)
    if size == 1:
        return x
    part_sum = [x[n] + x[n + size / 2] for n in xrange(size / 2)]
    part_min = [(x[n] - x[n + size / 2]) *
                _w(n, size) for n in xrange(size / 2)]
    part_sum = fft(part_sum)
    part_min = fft(part_min)
    result = [0] * size
    for i in xrange(size / 2):
        result[2 * i] = part_sum[i]
        result[2 * i + 1] = part_min[i]
    return result


def _ifft(x):
    size = len(x)
    if size == 1:
        return x
    even = _ifft(x[0::2])
    odd = _ifft(x[1::2])
    return [even[n] + np.conj(_w(n, size)) * odd[n] for n in xrange(size / 2)] + \
           [even[n] - np.conj(_w(n, size)) * odd[n] for n in xrange(size / 2)]


def ifft(x):
    x = _ifft(x)
    return [i / len(x) for i in x]
