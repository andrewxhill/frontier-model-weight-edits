import unittest

import numpy as np
from numpy.testing import assert_allclose

from combined_fit import joint_ridge


def fixture():
    Xi = np.array([[2.0, 1.0], [0.0, -1.0], [3.0, 2.0]])
    Yi = np.array([[1.0, 3.0], [2.0, -1.0], [4.0, 0.0]])
    Xc = np.array([[1.0, 0.0], [-1.0, 2.0]])
    W0 = np.array([[0.7, -2.0], [1.0, 1.5]])
    wi = np.array([1.0, 3.0, 2.0])
    wc = np.array([4.0, 1.0])
    return Xi, Yi, Xc, W0, wi, wc


class JointRidgeTest(unittest.TestCase):
    def test_matches_augmented_reference_and_diagnostics(self):
        Xi, Yi, Xc, W0, wi, wc = fixture()
        rho, lam = 2.5, 0.2
        W, b, diagnostics = joint_ridge(Xi, Yi, Xc, W0, rho, lam, wi, wc)

        # Form the reference with the intercept as an explicit design column.
        X = np.vstack((Xi, Xc))
        Y = np.vstack((Yi, np.zeros((len(Xc), Yi.shape[1]))))
        weights = np.r_[wi / wi.sum(), rho * wc / wc.sum()] / (1 + rho)
        mean = weights @ X
        variance = weights @ ((X - mean) ** 2)
        scale = variance.mean()
        prior_rows = np.c_[np.eye(X.shape[1]), np.zeros(X.shape[1])]
        augmented_X = np.vstack((
            np.sqrt(weights)[:, None] * np.c_[X, np.ones(len(X))],
            np.sqrt(lam * scale) * prior_rows,
        ))
        augmented_Y = np.vstack((
            np.sqrt(weights)[:, None] * Y,
            np.sqrt(lam * scale) * W0,
        ))
        expected = np.linalg.solve(augmented_X.T @ augmented_X, augmented_X.T @ augmented_Y)

        assert_allclose(np.vstack((W, b)), expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(W.dtype, np.dtype("float64"))
        self.assertEqual(b.dtype, np.dtype("float64"))
        island_mse = np.average(((Xi @ W + b - Yi) ** 2).mean(axis=1), weights=wi)
        camera_mse = np.average(((Xc @ W + b) ** 2).mean(axis=1), weights=wc)
        self.assertAlmostEqual(diagnostics["island_mse"], island_mse)
        self.assertAlmostEqual(diagnostics["camera_mse"], camera_mse)
        self.assertAlmostEqual(diagnostics["scale"], scale)
        self.assertAlmostEqual(diagnostics["island_weight_total"], 1 / (1 + rho))
        self.assertAlmostEqual(diagnostics["camera_weight_total"], rho / (1 + rho))
        self.assertEqual(diagnostics["island_rows"], 3)
        self.assertEqual(diagnostics["camera_rows"], 2)
        self.assertEqual(diagnostics["features"], 2)
        self.assertEqual(diagnostics["outputs"], 2)

    def test_camera_mse_decreases_for_shared_direction_as_camera_weight_rises(self):
        X = np.array([[-1.0, 0.0], [1.0, 0.0]])
        Y = X[:, :1]
        prior = np.zeros((2, 1))
        camera_mses = [
            joint_ridge(X, Y, X, prior, rho)[2]["camera_mse"]
            for rho in (0.1, 1.0, 10.0)
        ]
        self.assertGreater(camera_mses[0], camera_mses[1])
        self.assertGreater(camera_mses[1], camera_mses[2])

    def test_separable_directions_recover_island_and_zero_camera_targets(self):
        Xi = np.array([[-1.0, 0.0], [1.0, 0.0]])
        Yi = np.array([[-2.0], [2.0]])
        Xc = np.array([[0.0, -1.0], [0.0, 1.0]])
        W, b, _ = joint_ridge(Xi, Yi, Xc, np.zeros((2, 1)), 1.0, lam=1e-8)

        assert_allclose(Xi @ W + b, Yi, atol=1e-7, rtol=0)
        assert_allclose(Xc @ W + b, 0.0, atol=1e-12, rtol=0)

    def test_row_order_does_not_change_fit(self):
        Xi, Yi, Xc, W0, wi, wc = fixture()
        expected_W, expected_b, _ = joint_ridge(Xi, Yi, Xc, W0, 2.0, wi=wi, wc=wc)
        island_order = [2, 0, 1]
        camera_order = [1, 0]
        W, b, _ = joint_ridge(
            Xi[island_order], Yi[island_order], Xc[camera_order], W0, 2.0,
            wi=wi[island_order], wc=wc[camera_order],
        )

        assert_allclose(W, expected_W, atol=1e-12, rtol=1e-12)
        assert_allclose(b, expected_b, atol=1e-12, rtol=1e-12)

    def test_replicated_domain_rows_with_split_weights_do_not_change_fit(self):
        Xi, Yi, Xc, W0, wi, wc = fixture()
        expected_W, expected_b, _ = joint_ridge(Xi, Yi, Xc, W0, 0.5, wi=wi, wc=wc)
        W, b, _ = joint_ridge(
            Xi, Yi, np.repeat(Xc, 3, axis=0), W0, 0.5,
            wi=wi, wc=np.repeat(wc / 3, 3),
        )

        assert_allclose(W, expected_W, atol=1e-12, rtol=1e-12)
        assert_allclose(b, expected_b, atol=1e-12, rtol=1e-12)

    def test_zero_weight_rows_are_ignored(self):
        Xi, Yi, Xc, W0, wi, wc = fixture()
        expected_W, expected_b, _ = joint_ridge(Xi[1:], Yi[1:], Xc, W0, 2.0, wi=wi[1:], wc=wc)
        wi[0] = 0.0

        W, b, _ = joint_ridge(Xi, Yi, Xc, W0, 2.0, wi=wi, wc=wc)

        assert_allclose(W, expected_W, atol=1e-12, rtol=1e-12)
        assert_allclose(b, expected_b, atol=1e-12, rtol=1e-12)

    def test_constant_features_keep_prior_and_fit_intercept(self):
        Xi = np.array([[2.0, 3.0], [2.0, 3.0]])
        Yi = np.array([[1.0, 3.0], [2.0, 5.0]])
        Xc = Xi[:1]
        W0 = np.array([[1.0, 2.0], [-2.0, 1.0]])
        W, b, diagnostics = joint_ridge(Xi, Yi, Xc, W0, 1.0)

        assert_allclose(W, W0, atol=1e-12, rtol=1e-12)
        assert_allclose(b, 0.5 * Yi.mean(axis=0) - Xi[0] @ W0, atol=1e-12, rtol=1e-12)
        self.assertEqual(diagnostics["scale"], 1e-12)

    def test_rejects_inconsistent_shapes_and_empty_dimensions(self):
        Xi, Yi, Xc, W0, wi, wc = fixture()
        args = dict(Xi=Xi, Yi=Yi, Xc=Xc, W0=W0, camera_weight=1.0, wi=wi, wc=wc)
        for changes in (
            dict(Xi=Xi[:, 0]),
            dict(Yi=Yi[:2]),
            dict(Yi=Yi[:, :0], W0=W0[:, :0]),
            dict(Xc=Xc[:, :1]),
            dict(W0=W0[:, :1]),
            dict(wi=wi[:, None]),
            dict(wc=wc[:1]),
            dict(Xi=Xi[:0], Yi=Yi[:0], wi=wi[:0]),
            dict(Xc=Xc[:0], wc=wc[:0]),
            dict(Xi=Xi[:, :0], Xc=Xc[:, :0], W0=W0[:0]),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                joint_ridge(**(args | changes))

    def test_rejects_nonfinite_inputs(self):
        Xi, Yi, Xc, W0, wi, wc = fixture()
        args = dict(Xi=Xi, Yi=Yi, Xc=Xc, W0=W0, camera_weight=1.0, lam=0.01, wi=wi, wc=wc)
        for name in ("Xi", "Yi", "Xc", "W0", "wi", "wc"):
            array = args[name].copy()
            array.flat[0] = np.nan
            with self.subTest(name=name), self.assertRaises(ValueError):
                joint_ridge(**(args | {name: array}))
        for name in ("camera_weight", "lam"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                joint_ridge(**(args | {name: np.inf}))

    def test_rejects_negative_and_empty_weight_mass(self):
        Xi, Yi, Xc, W0, wi, wc = fixture()
        args = dict(Xi=Xi, Yi=Yi, Xc=Xc, W0=W0, camera_weight=1.0, wi=wi, wc=wc)
        for changes in (
            dict(wi=[-1.0, 2.0, 1.0]),
            dict(wc=[0.0, 0.0]),
            dict(camera_weight=-0.5),
            dict(camera_weight=0.0),
            dict(lam=-0.5),
            dict(lam=0.0),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                joint_ridge(**(args | changes))
