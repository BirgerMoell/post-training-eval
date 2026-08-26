import json
import tempfile
import unittest
from pathlib import Path

from post_training_eval.checkpoints import CheckpointError, inspect_checkpoint, parse_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_parse_hf_and_megatron(self):
        ref = parse_checkpoint("hf://owner/model@abc123")
        self.assertEqual((ref.format, ref.location, ref.revision), ("hf_hub", "owner/model", "abc123"))
        self.assertEqual(parse_checkpoint("megatron:///scratch/model").format, "megatron")

    def test_local_hf_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3", "max_position_embeddings": 262144}))
            (root / "tokenizer.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"test")
            report = inspect_checkpoint(parse_checkpoint(str(root)))
            self.assertEqual(report["status"], "compatible")
            self.assertEqual(report["architecture"]["max_position_embeddings"], 262144)

    def test_unknown_reference_rejected(self):
        with self.assertRaises(CheckpointError):
            parse_checkpoint("not-a-checkpoint")

    def test_megatron_distcp_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "latest_checkpointed_iteration.txt").write_text("42\n")
            iteration = root / "iter_0000042"
            iteration.mkdir()
            (iteration / "__0_0.distcp").write_bytes(b"checkpoint")
            (iteration / ".metadata").write_bytes(b"metadata")
            report = inspect_checkpoint(parse_checkpoint(f"megatron://{root}"))
            self.assertEqual(report["status"], "conversion_required")
            self.assertEqual(report["iteration"], 42)
            self.assertTrue(report["checks"]["distributed_metadata"])


if __name__ == "__main__":
    unittest.main()
