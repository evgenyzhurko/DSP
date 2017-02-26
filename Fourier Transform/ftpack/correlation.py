from fft import fft, ifft
import numpy as np


def conv(y, z, n):
    result = np.zeros(n)
    for i in range(n):
        k = i
        for j in range(n):
            result[i] += y[j] * z[k]
            k -= 1
        result[i] /= n

    return result


def corr(y, z, n):
    result = np.zeros(n)
    for i in range(n):
        k = -i
        for j in range(n):
            result[i] += y[k] * z[j]
            k += 1
        result[i] /= n

    return result


def corr_fft(y, z, n):

    y_value = fft(y)
    z_value = fft(z)

    result = [y_value[i].conj() * z_value[i] / n for i in xrange(n)]

    return [i.real for i in ifft(result)]


def conv_fft(y, z, n):

    y_value = fft(y)
    z_value = fft(z)

    result = [y_value[i] * z_value[i] / n for i in xrange(n)]

    return [i.real for i in ifft(result)]
