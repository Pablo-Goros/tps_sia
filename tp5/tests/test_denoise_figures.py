"""Figure generation for exercise 1.b: aggregation rule and the published campaign."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.denoise_figures import aggregate, make_figures

CAMPAIGN = Path(__file__).resolve().parents[1]/'results'/'denoising'


class AggregateTests(unittest.TestCase):
    def test_mean_min_max_over_models_per_level(self):
        rows = [{'model': m, 'level': p, 'character_index': None, 'mean_pixel_errors': v}
                for m, values in {'a': (1.0, 3.0), 'b': (2.0, 7.0)}.items() for p, v in zip((0.05, 0.1), values)]
        # Per-character rows must never leak into the all-character summary.
        rows.append({'model': 'a', 'level': 0.05, 'character_index': 3, 'mean_pixel_errors': 99.0})
        result = aggregate(rows, ['a', 'b'], [0.05, 0.1], 'mean_pixel_errors')
        np.testing.assert_allclose(result['mean'], [1.5, 5.0])
        np.testing.assert_allclose(result['min'], [1.0, 3.0])
        np.testing.assert_allclose(result['max'], [2.0, 7.0])

    def test_missing_summary_row_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate([], ['a'], [0.05], 'mean_pixel_errors')


@unittest.skipUnless((CAMPAIGN/'evaluation'/'metrics.json').is_file(), 'published denoising campaign not available')
class PublishedCampaignTests(unittest.TestCase):
    def test_figures_are_written_without_touching_the_campaign(self):
        before = {p: p.stat().st_mtime_ns for p in CAMPAIGN.rglob('*') if p.is_file()}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'figures'
            manifest = make_figures(CAMPAIGN, output)
            for name in manifest['files']:
                self.assertGreater((output/name).stat().st_size, 1000, name)
            self.assertEqual(sum(name.startswith('samples-p') for name in manifest['files']), 5)
            saved = json.loads((output/'figures-manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['example_indices'], manifest['example_indices'])
            with self.assertRaises(ValueError):  # outputs are never overwritten
                make_figures(CAMPAIGN, output)
        after = {p: p.stat().st_mtime_ns for p in CAMPAIGN.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_incomplete_campaign_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            make_figures(temporary, Path(temporary)/'figures')


if __name__ == '__main__':
    unittest.main()
