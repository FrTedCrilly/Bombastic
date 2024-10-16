import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# each time we change the cython file we need to redo the set up process so it gets reflected in the new module.


def apply_zscore(np.ndarray[np.float64_t, ndim=2] var, int Zwin, bint Zexpand, bint rmMean=True, bint cheat=True):
    cdef Py_ssize_t n_rows = var.shape[0]
    cdef Py_ssize_t n_cols = var.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.empty_like(var)
    cdef np.ndarray[np.float64_t, ndim=1] window
    cdef double mean, std
    cdef int i, j

    for j in range(n_cols):
        for i in range(n_rows):
            if Zexpand:
                window = var[:i+1, j]
            else:
                start = max(0, i - Zwin + 1)
                window = var[start:i+1, j]

            if i < Zwin:
                if cheat:
                    # Use the first Zwin elements for std (and mean if rmMean=True)
                    window = var[:Zwin, j]
                    mean = window.mean() if rmMean else 0.0
                    std = window.std()
                else:
                    result[i, j] = np.nan
                    continue
            else:
                mean = window.mean() if rmMean else 0.0
                std = window.std()

            result[i, j] = (var[i, j] - mean) / std if std != 0 else 0.0

    return result

def apply_zscoreOld(np.ndarray[np.float64_t, ndim=2] var, int Zwin, bint Zexpand, bint rmMean=True, bint cheat=True):
    cdef Py_ssize_t n_rows = var.shape[0]
    cdef Py_ssize_t n_cols = var.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.empty_like(var)
    cdef np.ndarray[np.float64_t, ndim=1] window
    cdef double mean, std, sum, sum_sq
    cdef int i, j, k

    for j in range(n_cols):
        for i in range(n_rows):
            if Zexpand:
                window = var[:i+1, j]
            else:
                start = max(0, i - Zwin + 1)
                window = var[start:i+1, j]

            if rmMean:
                mean = window.mean()
                std = window.std()
            else:
                sum = window.sum()
                sum_sq = (window**2).sum()
                mean = sum / len(window)
                std = sqrt((sum_sq - sum * mean) / len(window))

            result[i, j] = (var[i, j] - mean) / std if std != 0 else 0.0

    if cheat:
        for j in range(n_cols):
            first_valid = np.argmax(~np.isnan(result[:, j]))
            result[:, j] = np.nan_to_num(result[:, j], nan=result[first_valid, j])

    return result

def apply_quantile(np.ndarray[np.float64_t, ndim=2] var, int Qwin, bint Qexpand, bint cheat=True):
    cdef Py_ssize_t n_rows = var.shape[0]
    cdef Py_ssize_t n_cols = var.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.empty_like(var)
    cdef np.ndarray[np.float64_t, ndim=1] window
    cdef int i, j
    cdef int start

    for j in range(n_cols):
        for i in range(n_rows):
            if Qexpand:
                window = var[:i+1, j]
            else:
                start = max(0, i - Qwin + 1)
                window = var[start:i+1, j]

            if i < Qwin:
                if cheat:
                    # Use the first Qwin elements for the quantile calculation
                    window = var[:Qwin, j]
                    result[i, j] = np.sum(window <= var[i, j]) / len(window)
                else:
                    result[i, j] = np.nan
            else:
                result[i, j] = np.sum(window <= var[i, j]) / len(window)

    return result
def apply_quantileOld(np.ndarray[np.float64_t, ndim=2] var, int Qwin, bint Qexpand):
    cdef Py_ssize_t n_rows = var.shape[0]
    cdef Py_ssize_t n_cols = var.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.empty_like(var)
    cdef np.ndarray[np.float64_t, ndim=1] window
    cdef int i, j

    for j in range(n_cols):
        for i in range(n_rows):
            if Qexpand:
                window = var[:i+1, j]
            else:
                start = max(0, i - Qwin + 1)
                window = var[start:i+1, j]

            result[i, j] = np.sum(window <= var[i, j]) / len(window)

    return result

def apply_sand(np.ndarray[np.float64_t, ndim=2] signal, double sand):
    cdef Py_ssize_t n = signal.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] adjusted_signal = np.empty_like(signal)
    cdef double prev_value = signal[0]
    cdef double curr_value
    cdef Py_ssize_t i

    adjusted_signal[0] = prev_value

    for i in range(1, n):
        curr_value = signal[i]
        if abs(curr_value - prev_value) >= sand:
            adjusted_signal[i] = curr_value
            prev_value = curr_value
        else:
            adjusted_signal[i] = prev_value

    return adjusted_signal

def sigSandOld(np.ndarray[np.float64_t, ndim=2] quantiles, double pEnter, double pExit):
    cdef Py_ssize_t n = quantiles.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] signals = np.zeros(n, dtype=np.float64)
    cdef double long_enter = 1.0 - pEnter
    cdef double long_exit = 1.0 - pExit
    cdef double short_enter = pEnter
    cdef double short_exit = pExit
    cdef int position = 0  # 1 for long, -1 for short, 0 for no position
    cdef double quantile
    cdef Py_ssize_t i

    for i in range(n):
        quantile = quantiles[i]

        if position == 0:
            if quantile > long_enter:
                position = 1
                signals[i] = 1
            elif quantile < short_enter:
                position = -1
                signals[i] = -1
        elif position == 1:
            if quantile < long_exit:
                position = 0
                signals[i] = 0
            else:
                signals[i] = 1
        elif position == -1:
            if quantile > short_exit:
                position = 0
                signals[i] = 0
            else:
                signals[i] = -1

    return signals

def sigSand(np.ndarray[np.float64_t, ndim=2] quantiles, double pEnter, double pExit):
    cdef Py_ssize_t n = quantiles.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] signals = np.empty_like(quantiles)  # Using np.empty_like
    cdef double long_enter = 1.0 - pEnter
    cdef double long_exit = 1.0 - pExit
    cdef double short_enter = pEnter
    cdef double short_exit = pExit
    cdef int position = 0  # 1 for long, -1 for short, 0 for no position
    cdef double quantile
    cdef Py_ssize_t i

    for i in range(n):
        quantile = quantiles[i, 0]  # Accessing the single column in the 2D array

        if position == 0:
            if quantile > long_enter:
                position = 1
                signals[i, 0] = 1
            elif quantile < short_enter:
                position = -1
                signals[i, 0] = -1
            else:
                signals[i, 0] = 0  # Explicitly set to 0 if no position
        elif position == 1:
            if quantile < long_exit:
                position = 0
                signals[i, 0] = 0
            else:
                signals[i, 0] = 1
        elif position == -1:
            if quantile > short_exit:
                position = 0
                signals[i, 0] = 0
            else:
                signals[i, 0] = -1

    return signals


def continousSig(double value):
    if (value >= 0.45) and (value <= 0.55):
        return 0.0
    else:
        return np.floor(value * 20) / 20

def apply_continousSig(np.ndarray[np.float64_t, ndim=2] series):
    cdef int n = series.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.empty(n)
    cdef int i

    for i in range(n):
        result[i] = continousSig(series[i][1])

    return result



