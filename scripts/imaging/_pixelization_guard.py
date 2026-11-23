def assert_log_det_terms(label, inversion):
    # The evidence correction can cancel even when both determinants are large.
    assert inversion is not None, f"{label}: expected a pixelization inversion"
    for name in (
        "log_det_curvature_reg_matrix_term",
        "log_det_regularization_matrix_term",
    ):
        value = float(getattr(inversion, name))
        assert abs(value) >= 1.0, f"{label}: inversion.{name} ({value}) is below 1 nat"
