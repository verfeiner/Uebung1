# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math
import matplotlib.pyplot as plt

alpha = 2
beta = 3
a = 0
b = 1
esp= 0.001
def f(x, alpha, beta):
    return math.exp(-alpha * x) + x ** beta


while (b - a) > esp:
    # Trisect the interval
    mid = (b - a) / 2
    fl=f(mid-esp, alpha, beta)
    fr=f(mid+esp, alpha, beta)
    if f(mid-esp) < f(mid+esp):
        a = mid
    else:
        b = mid

x_min = (a + b) / 2
min_value = f(x_min, alpha, beta)

print(f"The minimum of f(x) on [{a}, {b8}] with alpha={alpa} and beta={beta} is at x = { }, f(x) = {min_value}")
plt.plot(x_min,min_value)
plt.show()
