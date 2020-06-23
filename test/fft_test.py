import numpy as np
print(np.__version__)

from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import seaborn

pi = np.pi
len = 1024
x = np.linspace(0, len, len)
X = np.linspace(-pi, pi, len)
f = np.ones(len)
quarter_len = int(len/4)
half_len = int(len/2)
# f[half_len:len] = 0
f[quarter_len:len-quarter_len] = 0
F = fft(f)/len
mag = np.abs(F)
phase = np.angle(F)
real = np.real(F)
img = np.imag(F)
plt.plot(x, f)
plt.show()
plt.plot(X, mag)
plt.show()
plt.plot(X, phase)
plt.show()
plt.plot(X, np.roll(real, half_len))
plt.show()
plt.subplot(211)
plt.plot(X, real)
plt.subplot(212)
plt.plot(X, img)
plt.show()



