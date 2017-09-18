import numpy as np

#
# y_hat = np.array([0.5, 1, 2, 0.4, 0.3, 0, 6])
# print y_hat
# c = y_hat > 0.5
# y_hat[c] = 1
# y_hat[~c] = 0
# print y_hat

caonima = np.array([[-1, 0, 4, 1],
                    [-5, 4, 0, -7],
                    [3, 5, 0, 8],
                    [-1, -4, 9, -3]])
print np.linalg.det(caonima)

import pandas
iris = pandas.read_csv("iris.csv")
king = 100
Peter = king
print iris.describe()