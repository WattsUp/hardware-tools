"""PAM2 helper functions definitions, see measurement.eyediagram.pam2
"""

cimport numpy as np

cdef dict sample_vertical_c(
    np.ndarray[np.float64_t, ndim=1] waveform_y,
    np.ndarray[np.float64_t, ndim=1] centers_t,
    np.ndarray[np.int32_t, ndim=1] centers_i,
    np.float64_t t_delta,
    np.float64_t t_sym,
    np.float64_t y_half,
    np.float64_t level_width,
    np.float64_t cross_width)

cdef dict sample_horizontal_c(
    np.ndarray[np.float64_t, ndim=1] waveform_y,
    np.ndarray[np.float64_t, ndim=1] centers_t,
    np.ndarray[np.int32_t, ndim=1] centers_i,
    np.ndarray[np.int8_t, ndim=1] edge_dir,
    np.float64_t t_delta,
    np.float64_t t_sym,
    np.float64_t y_zero,
    np.float64_t y_ua,
    np.float64_t y_cross,
    np.float64_t hist_height,
    np.float64_t edge_lower,
    np.float64_t edge_upper)
