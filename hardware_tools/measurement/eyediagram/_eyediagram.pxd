"""Eye Diagram helper functions definitions, see measurement.eyediagram.eyediagram
"""

cimport numpy as np

cdef dict sample_mask_c(
    np.ndarray[np.float64_t, ndim=1] waveform_y,
    list centers_t,
    list centers_i,
    np.float64_t t_delta,
    np.float64_t t_sym,
    np.float64_t y_zero,
    np.float64_t y_ua,
    list mask_paths,
    list mask_margins)

cdef list y_slice_c(
    np.ndarray[np.float64_t, ndim=1] waveform_y,
    list centers_t,
    list centers_i,
    np.float64_t t_delta,
    np.float64_t t_sym,
    np.float64_t y_zero,
    np.float64_t y_ua,
    list y_slices)

cdef np.ndarray[np.float64_t, ndim=2] stack_c(
    np.ndarray[np.float64_t, ndim=1] waveform_y,
    list centers_t,
    list centers_i,
    np.float64_t t_delta,
    np.float64_t t_sym,
    np.float64_t min_y,
    np.float64_t max_y,
    int resolution,
    bint point_cloud)
