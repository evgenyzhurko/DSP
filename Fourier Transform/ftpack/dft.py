import numpy as np


def dft(x):
    size = len(x)
    a = np.zeros(size, dtype=np.complex)
    for k in xrange(size):
        for j in xrange(size):
            a[k] += x[j] * np.exp(-1j * 2 * np.pi * k * j / size)
    return a


def idft(x):
    size = len(x)
    a = np.zeros(size, dtype=np.complex)
    for j in xrange(size):
        for i in range(size):
            a[j] += x[i] * np.exp(1j * 2 * np.pi * i * j / size)
    return a / size
