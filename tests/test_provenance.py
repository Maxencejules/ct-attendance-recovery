import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401
from ci.prove_attendance import prove


class ProvenanceTests(unittest.TestCase):
    def test_repeated_output_manifest_hashes_current_files_without_self_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'unrelated.txt').write_text('Not generated evidence')
            with contextlib.redirect_stdout(io.StringIO()):
                prove(out)
                first = (out / 'proof.json').read_bytes()
                prove(out)
            self.assertEqual(first, (out / 'proof.json').read_bytes())
            proof = json.loads(first)
            self.assertNotIn('proof.json', proof['output_sha256'])
            self.assertNotIn('unrelated.txt', proof['output_sha256'])
            self.assertEqual(len(proof['output_sha256']), 18)
            for name, digest in proof['output_sha256'].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
