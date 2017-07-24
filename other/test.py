import numpy as np

y_hat = np.array([0.5, 1, 2, 0.4, 0.3, 0, 6])
print y_hat
c = y_hat > 0.5
y_hat[c] = 1
y_hat[~c] = 0
print y_hat
