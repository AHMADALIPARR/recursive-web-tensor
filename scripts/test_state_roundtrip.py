"""Verify hash-sealed checkpoint bytes survive Git on Linux and Windows."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


class StateRoundtrip(unittest.TestCase):
    def test_checkpoint_hash_survives_checkout_modes(self):
        with tempfile.TemporaryDirectory(prefix="rwt-git-roundtrip-") as temp:
            seed = Path(temp) / "seed"
            seed.mkdir()
            git(seed, "init", "-b", "main")
            git(seed, "config", "user.name", "Local workflow validation")
            git(seed, "config", "user.email", "local-test@example.invalid")
            (seed / ".gitignore").write_bytes(b"/state\n/checkpoints\n")
            shutil.copyfile(PROJECT / ".gitattributes", seed / ".gitattributes")
            manifest = seed / "checkpoints" / "v1" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            payload = b'{\n  "version": "v1",\n  "status": "promoted"\n}\n'
            manifest.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            state = seed / "state"
            state.mkdir()
            (state / "current.json").write_bytes(
                (json.dumps({"version": "v1", "manifest_sha256": digest}) + "\n").encode()
            )
            git(seed, "add", ".gitignore", ".gitattributes")
            # These paths deliberately remain ignored for ordinary local runs.
            self.assertNotEqual(subprocess.run(
                ["git", "-C", str(seed), "add", "state", "checkpoints"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).returncode, 0)
            git(seed, "add", "-f", "state", "checkpoints")
            git(seed, "commit", "-m", "Seal checkpoint fixture")
            for mode in ("true", "false", "input"):
                with self.subTest(autocrlf=mode):
                    restored = Path(temp) / ("restored-" + mode)
                    subprocess.run(
                        ["git", "clone", "--no-checkout", str(seed), str(restored)],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    git(restored, "config", "core.autocrlf", mode)
                    git(restored, "checkout", "main")
                    actual = (restored / "checkpoints" / "v1" / "manifest.json").read_bytes()
                    pointer = json.loads((restored / "state" / "current.json").read_bytes())
                    self.assertEqual(actual, payload)
                    self.assertEqual(hashlib.sha256(actual).hexdigest(), pointer["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
