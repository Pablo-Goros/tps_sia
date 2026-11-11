import unittest
from src.denoise_training import select_runs, validation_score

class SelectionTests(unittest.TestCase):
    def test_ranking_uses_validation_seed_means_and_rejects_incomplete_campaigns(self):
        rows = [{'run_id': f'{variant}-{seed}', 'variant': variant, 'seed': seed,
                 'status': 'completed', 'best': {'score': [error, .01], 'epoch': 100}}
                for variant, errors in [('winner', [1, 5, 3]), ('latent-10', [2, 2, 2])]
                for seed, error in zip([42,43,44], errors)]
        selected = select_runs(rows, ['winner', 'latent-10'], [42,43,44])
        self.assertEqual(selected['variant'], 'latent-10')
        self.assertEqual(selected['run_id'], 'latent-10-42')
        with self.assertRaises(ValueError):
            select_runs(rows[:-1], ['winner','latent-10'], [42,43,44])
        points = [{'level': level, 'mean_pixel_errors': error, 'mse_per_pixel': .1}
                  for level, error in [(0,99),(.05,1),(.1,2),(.2,3),(.3,99)]]
        self.assertEqual(validation_score(points, [.05,.1,.2])[0], 2.)
        self.assertAlmostEqual(validation_score(points, [.05,.1,.2])[1], .1)
