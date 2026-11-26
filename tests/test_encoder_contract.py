"""Contract and end-to-end tests for the NumPy image encoder.

Run from the repository root:

    python tests/test_encoder_contract.py

These tests complement check.py. They use relationships (single vs batch,
permutation, duplicates, determinism) and synthetic images rather than
copying expected arrays from the implementation.
"""
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from image_encoder import ImageEncoder, MultiHeadAttention  # noqa: E402
from runtime import load_images, load_weights  # noqa: E402

WEIGHTS = ROOT / 'weights' / 'image_encoder.npz'
REFERENCE = ROOT / 'assets' / 'reference_outputs.npz'
IMAGES = ROOT / 'assets' / 'images'


class TestPatchAndHeads(unittest.TestCase):
    def test_patchify_patch_and_channel_order(self):
        encoder = ImageEncoder.__new__(ImageEncoder)
        grid = np.arange(3 * 64 * 64, dtype=np.float32).reshape(1, 3, 64, 64)
        patches = encoder.patchify(grid)
        self.assertEqual(patches.shape, (1, 4, 3072))
        for row in range(2):
            for column in range(2):
                expected = np.concatenate([
                    grid[0, channel,
                         row * 32:(row + 1) * 32,
                         column * 32:(column + 1) * 32].reshape(-1)
                    for channel in range(3)
                ])
                np.testing.assert_array_equal(
                    patches[0, row * 2 + column], expected)

    def test_split_heads_preserves_token_and_head_order(self):
        attention = MultiHeadAttention.__new__(MultiHeadAttention)
        attention.heads = 4
        attention.head_dim = 2
        attention.width = 8
        values = np.arange(2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)
        heads = attention.split_heads(values)
        self.assertEqual(heads.shape, (2, 4, 3, 2))
        for batch in range(2):
            for head in range(4):
                for token in range(3):
                    expected = values[batch, token, head * 2:(head + 1) * 2]
                    np.testing.assert_array_equal(
                        heads[batch, head, token], expected)


class TestEncoderContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not WEIGHTS.exists():
            raise unittest.SkipTest('weights/image_encoder.npz is not present')
        cls.encoder = ImageEncoder(load_weights())
        cls.images = load_images()

    def test_output_contract_for_batch_sizes(self):
        for batch in (1, 2, 5):
            output = self.encoder(self.images[:batch])
            self.assertEqual(output.shape, (batch, 512))
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(np.isfinite(output).all())
            np.testing.assert_allclose(
                np.linalg.norm(output, axis=1), 1.0, atol=2e-4)

    def test_batch_invariance_and_order(self):
        base = self.encoder(self.images[:3])
        single = self.encoder(self.images[:1])[0]
        np.testing.assert_allclose(single, base[0], atol=2e-6, rtol=2e-5)

        order = [2, 0, 1]
        permuted = self.encoder(self.images[order])
        np.testing.assert_allclose(
            permuted, base[order], atol=2e-6, rtol=2e-5)

        duplicate = self.encoder(self.images[[1, 1]])
        np.testing.assert_allclose(
            duplicate[0], duplicate[1], atol=2e-6, rtol=0)

    def test_deterministic(self):
        first = self.encoder(self.images[:2])
        second = self.encoder(self.images[:2])
        np.testing.assert_array_equal(first, second)


class TestCommandLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not WEIGHTS.exists():
            raise unittest.SkipTest('weights/image_encoder.npz is not present')

    def run_cli(self, images, output):
        command = [
            sys.executable, str(ROOT / 'run.py'), '--images',
            *map(str, images), '--output', str(output),
        ]
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        with np.load(output) as data:
            return data['ids'].tolist(), data['output_tokens']

    def test_ids_and_rows_follow_input_order(self):
        output = ROOT / ('.test-output-' + uuid.uuid4().hex + '.npz')
        try:
            ids, tokens = self.run_cli(
                [IMAGES / 'image_02.jpg', IMAGES / 'image_01.jpg'], output)
            self.assertEqual(ids, ['image_02', 'image_01'])
            self.assertEqual(tokens.shape, (2, 512))
            self.assertEqual(tokens.dtype, np.float32)
            self.assertTrue(np.isfinite(tokens).all())
            np.testing.assert_allclose(
                np.linalg.norm(tokens, axis=1), 1.0, atol=2e-4)
        finally:
            output.unlink(missing_ok=True)

    def test_runs_without_reference_file(self):
        if not REFERENCE.exists():
            self.skipTest('assets/reference_outputs.npz is not present')
        backup = REFERENCE.with_name('reference_outputs.npz.audit-bak')
        if backup.exists():
            backup.unlink()
        REFERENCE.rename(backup)
        output = ROOT / ('.test-output-' + uuid.uuid4().hex + '.npz')
        try:
            ids, tokens = self.run_cli([IMAGES / 'image_01.jpg'], output)
            self.assertEqual(ids, ['image_01'])
            self.assertEqual(tokens.shape, (1, 512))
            self.assertTrue(np.isfinite(tokens).all())
        finally:
            output.unlink(missing_ok=True)
            backup.rename(REFERENCE)


if __name__ == '__main__':
    unittest.main(verbosity=2)
