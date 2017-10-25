from scipy.special import comb, perm

s = 0.0
for i in xrange(5 + 1, 11):
    s += comb(10, i) * 0.3 ** i * 0.7 ** (10 - i)
print s
