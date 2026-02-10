import numpy as np


def zf_precoder(H):
    # H: [K, M] complex
    K, M = H.shape
    if K > M:
        Hh = np.conjugate(H.T)
        W = Hh @ np.linalg.pinv(H @ Hh)
    else:
        W = np.conjugate(H.T) @ np.linalg.pinv(H @ np.conjugate(H.T))
    # Normalize total power to 1
    power = np.sum(np.linalg.norm(W, axis=0) ** 2)
    if power > 0:
        W = W / np.sqrt(power)
    return W  # [M, K]


def sum_rate(H_true, H_est, snr_db=10.0):
    # H_true, H_est: [T, K, M] complex
    snr = 10 ** (snr_db / 10.0)
    T, K, M = H_true.shape
    rates = np.zeros(T)

    for t in range(T):
        Ht = H_true[t]
        He = H_est[t]
        W = zf_precoder(He)
        signal = np.zeros(K)
        interference = np.zeros(K)

        for k in range(K):
            hk = Ht[k].reshape(1, M)
            wk = W[:, k].reshape(M, 1)
            signal[k] = np.abs(hk @ wk) ** 2
            for j in range(K):
                if j == k:
                    continue
                wj = W[:, j].reshape(M, 1)
                interference[k] += np.abs(hk @ wj) ** 2

        sinr = (snr * signal) / (1.0 + snr * interference)
        rates[t] = np.sum(np.log2(1 + sinr.real))

    return rates
