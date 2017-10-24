import numpy as np
from ABS import *
if __name__ == "__main__":
    bound = np.tile([[-600], [600]], 25)
    abs = ABS(60, 25, bound, 1000, [100,  0.5])
    abs.solve()