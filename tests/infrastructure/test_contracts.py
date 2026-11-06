"""Meaningful configuration checks that run without the future scientific stack."""
from dataclasses import FrozenInstanceError, asdict
import json
import unittest

from doe_core.contracts import OpticsConfig


class ConfigTests(unittest.TestCase):
    def make(self, **changes):
        values = dict(wavelength_m=532e-9, pixel_pitch_m=(8e-6, 9e-6), distance_m=0.1)
        values.update(changes)
        return OpticsConfig(**values)

    def test_valid_config_and_zero_distance(self):
        config = self.make(distance_m=0)
        self.assertEqual(config.pixel_pitch_m, (8e-6, 9e-6))
        self.assertEqual(config.pad_factor, 1)
        self.assertEqual(config.distance_m, 0)

    def test_immutable_config(self):
        with self.assertRaises(FrozenInstanceError):
            self.make().distance_m = 2

    def test_config_serializes_without_nonfinite_numbers(self):
        decoded = json.loads(json.dumps(asdict(self.make()), allow_nan=False))
        self.assertEqual(decoded['pixel_pitch_m'], [8e-6, 9e-6])

    def test_invalid_wavelength(self):
        for value in (0, -1, float('nan'), float('inf'), True, '532'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make(wavelength_m=value)

    def test_invalid_pitch_shape_and_values(self):
        for value in ([1, 1], (1,), (1, 2, 3), (0, 1), (1, float('nan')), (True, 1)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make(pixel_pitch_m=value)

    def test_invalid_distance(self):
        for value in (-1, float('nan'), float('inf'), True, '1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make(distance_m=value)

    def test_invalid_padding(self):
        for value in (0, -1, 2.0, True, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make(pad_factor=value)

    def test_explicit_padding(self):
        self.assertEqual(self.make(pad_factor=2).pad_factor, 2)


if __name__ == '__main__':
    unittest.main()
