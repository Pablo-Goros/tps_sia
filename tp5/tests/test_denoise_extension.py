"""Extension of the 1.b campaign: guards, resume continuity and untouched published results."""
import json
from pathlib import Path
import tempfile
import unittest

from src.denoise_extension import DOCUMENTATION, extend_campaign, make_figures, protocol_text, tree_hashes

CAMPAIGN = Path(__file__).resolve().parents[1]/'results'/'denoising'
AVAILABLE = (CAMPAIGN/'selection.json').is_file() and (CAMPAIGN/'runs'/'latent-10-seed-44'/'last_model.npz').is_file()


class ProtocolTests(unittest.TestCase):
    def test_protocol_text_states_budgets_and_that_it_is_exploratory(self):
        text = protocol_text(3000, 6000)
        self.assertIn('exploratorio', text)
        self.assertIn('3000', text)
        self.assertIn('6000', text)
        self.assertIn('no se modifican', text)


@unittest.skipUnless(AVAILABLE, 'published denoising campaign not available')
class PublishedCampaignTests(unittest.TestCase):
    def test_short_extension_keeps_the_published_campaign_and_never_worsens_the_best(self):
        before = tree_hashes(CAMPAIGN, DOCUMENTATION)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'extension'
            info = extend_campaign(CAMPAIGN, output, end_epoch=3100, log=lambda text: None)
            self.assertEqual(info['start_epoch'], 3000)
            self.assertEqual(info['end_epoch'], 3100)
            selection = json.loads((output/'selection-3100.json').read_text(encoding='utf-8'))
            self.assertEqual(len(selection['runs']), 6)
            for run in selection['runs']:
                published = run['best_3000']
                # The best checkpoint can only improve: the published best is kept unless beaten.
                self.assertLessEqual((run['best']['score'], run['best']['epoch']), (published['score'], published['epoch']))
                history = json.loads((output/'runs'/run['run_id']/'history.json').read_text(encoding='utf-8'))['validation']
                self.assertEqual([point['epoch'] for point in history][-2:], [3000, 3100])
                self.assertEqual(run['epochs_completed'], 3100)
            self.assertTrue((output/'comparison-3000-vs-3100.csv').is_file())
            self.assertTrue((output/'by-level-3100.csv').is_file())
            names = make_figures(output, CAMPAIGN)
            self.assertEqual(len(names), 2)
            for name in names:
                self.assertGreater((output/'figures'/name).stat().st_size, 1000)
            with self.assertRaises(ValueError):          # outputs are never overwritten
                extend_campaign(CAMPAIGN, output, end_epoch=3100, log=lambda text: None)
        self.assertEqual(before, tree_hashes(CAMPAIGN, DOCUMENTATION))   # no published data was touched

    def test_invalid_budgets_are_rejected_before_creating_anything(self):
        with tempfile.TemporaryDirectory() as temporary:
            for end in (3000, 2000, 3050):
                with self.assertRaises(ValueError):
                    extend_campaign(CAMPAIGN, Path(temporary)/f'ext-{end}', end_epoch=end, log=lambda text: None)
                self.assertFalse((Path(temporary)/f'ext-{end}').exists())


if __name__ == '__main__':
    unittest.main()
