import matplotlib.pyplot as plt

import numpy as np

W = np.arange(1, 10, 0.1)
gammas = np.arange(-5, 0, 0.3)
# gammas=np.delete(gammas, 3)
gammas
# U=W**gamma/gamma

for gamma in gammas:
    plt.plot(W, W ** gamma / gamma, label=gamma)
    # plt.plot(W, np.log(W))
# plt.plot(W, np.log(W))
plt.plot(W, np.log(W), label='log(W)')
plt.legend()
plt.show()
