from ftpack.correlation import corr_fft, conv_fft, conv, corr

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf

N = 16


def y(x):
    return np.sin(2 * x)


def z(x):
    return np.cos(7 * x)


y_val = [y(2 * np.pi * i / N) for i in xrange(N)]
z_val = [z(2 * np.pi * i / N) for i in xrange(N)]


correlation = corr(y_val, z_val, N)

correlation_fft = corr_fft(y_val, z_val, N)

convolution = conv(y_val, z_val, N)

convolution_fft = conv_fft(y_val, z_val, N)

with pdf.PdfPages('results.pdf') as file:

    plt.plot(np.arange(N), y_val)
    plt.title("Y function")
    plt.xlabel('x')
    plt.ylabel('y=sin(2x)')
    file.savefig()
    plt.close()

    plt.plot(np.arange(N), z_val)
    plt.title("Z function")
    plt.xlabel('x')
    plt.ylabel('z=cos(7x)')
    file.savefig()
    plt.close()

    plt.plot(np.arange(N), y_val)
    plt.plot(np.arange(N), z_val)
    plt.title("Y + Z function")
    plt.xlabel('x')
    plt.ylabel('y')
    file.savefig()
    plt.close()

    plt.stem(np.arange(N), correlation, asevlines=True)
    plt.title("Correlation")
    plt.xlabel('x')
    plt.ylabel('y')
    file.savefig()
    plt.close()

    plt.stem(np.arange(N), correlation_fft, asevlines=True)
    plt.title("Correlation FFT")
    plt.xlabel('x')
    plt.ylabel('y')
    file.savefig()
    plt.close()

    plt.stem(np.arange(N), convolution, asevlines=True)
    plt.title("Convolution")
    plt.xlabel('x')
    plt.ylabel('y')
    file.savefig()
    plt.close()

    plt.stem(np.arange(N), convolution_fft, asevlines=True)
    plt.title("Convolution FFT")
    plt.xlabel('x')
    plt.ylabel('y')
    file.savefig()
    plt.close()
