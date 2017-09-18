import math
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = np.arange(0.05, 3, 0.05)
    print x
    y1 = [math.log(a, 1.5) for a in x]
    plt.plot(x, y1, linewidth=2, color='#007500', label='log1.5(x)')
    plt.plot([1, 1], [y1[0], y1[-1]], "r--", linewidth=2)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
