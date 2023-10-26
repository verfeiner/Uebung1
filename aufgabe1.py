# This is a sample Python script.
import numpy as np
import math
import matplotlib.pyplot as plt

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
alpha=1
beta = 2

esp = 0.001
while   alpha<6:
        a = 0
        b = 1
        def f(x, alpha, beta):
            return math.exp(-alpha * x) + x ** beta


        while (b - a) > esp:
            # Trisect the interval
            tri = (b - a) / 3
            fl = f(a + tri, alpha, beta)
            fr = f(b - tri, alpha, beta)

            if fl < fr:
                b = b - tri
            else:
                a = a + tri

        x_min = (a + b) / 2
        min_value = f(x_min, alpha, beta)

        x = np.arange(-2, 2, 0.1)
        y = []
        for t in x:
            y1 = math.exp(-alpha * t) + t ** beta
            y.append(y1)

        print(f"The minimum of f(x) on [0, 1] with alpha={alpha} and beta={beta} is at x = {x_min}, f(x) = {min_value}")  # f bedeutet in ""kannst du auch vekor benutzen.
        plt.plot(x, y, label=f"alpha={alpha} and beta={beta}")
        plt.scatter(x_min, min_value,label="minimum", marker="o")

        alpha=alpha+1

plt.ylim(-0, 4)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title(f"convex funtion with torelance={esp}")
plt.legend()  # 'makieren'
plt.show()
