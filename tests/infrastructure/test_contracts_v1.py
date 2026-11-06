"""Case constants/geometry validation only; these are not Fresnel physics tests."""
from dataclasses import FrozenInstanceError
import unittest
from doe_core.contracts import OpticsConfig
from doe_core.contracts_v1 import CaseConfigV1, FresnelGeometry


class V1ContractTests(unittest.TestCase):
    def test_official_case_and_distance_order(self):
        c = CaseConfigV1()
        self.assertEqual(c.image_shape, (128, 128))
        self.assertEqual(c.input_pitch_m, (10e-6, 10e-6))
        self.assertEqual(c.distances_m, (0.0241, 0.0253, 0.0265, 0.0277))
        self.assertEqual(c.wavelength_m, 532e-9)

    def test_native_output_sampling_all_planes(self):
        expected_um = (10.0165625, 10.5153125, 11.0140625, 11.5128125)
        for i, expected in enumerate(expected_um):
            dy, dx = CaseConfigV1().plane(i).output_pitch_m(128, 128)
            self.assertAlmostEqual(dy * 1e6, expected, places=8)
            self.assertEqual(dy, dx)

    def test_rectangular_anisotropic_geometry(self):
        dy, dx = FresnelGeometry(500e-9, (2e-6, 4e-6), 0.02).output_pitch_m(5, 7)
        self.assertAlmostEqual(dy, 0.001)
        self.assertAlmostEqual(dx, 1e-8 / 28e-6)

    def test_zero_and_invalid_distance_rejected(self):
        for z in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(z=z), self.assertRaises(ValueError):
                FresnelGeometry(532e-9, (10e-6, 10e-6), z)

    def test_invalid_wavelength_and_pitch(self):
        for wave, pitch in ((0,(1,1)), (True,(1,1)), (1,[1,1]), (1,(1,)), (1,(0,1)), (1,(1,float('nan')))):
            with self.subTest(wave=wave,pitch=pitch), self.assertRaises(ValueError):
                FresnelGeometry(wave, pitch, 1)

    def test_case_is_immutable_and_indices_strict(self):
        c = CaseConfigV1()
        with self.assertRaises(FrozenInstanceError):
            c.wavelength_m = 1
        for i in (-1, 4, True, 1.0):
            with self.subTest(i=i), self.assertRaises(ValueError): c.plane(i)

    def test_invalid_grid_shape(self):
        g = CaseConfigV1().plane(0)
        for h,w in ((1,128),(128,0),(True,128),(128,2.0)):
            with self.subTest(h=h,w=w), self.assertRaises(ValueError): g.output_pitch_m(h,w)

    def test_v0_still_has_zero_distance_identity_geometry(self):
        self.assertEqual(OpticsConfig(532e-9, (10e-6,10e-6), 0).distance_m, 0)
