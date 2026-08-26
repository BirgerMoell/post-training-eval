from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / ".agents" / "skills" / "post-training-eval"


class AgentSkillTests(unittest.TestCase):
    def test_canonical_skill_and_claude_wrapper_are_connected(self) -> None:
        skill = (CANONICAL / "SKILL.md").read_text()
        wrapper = (ROOT / ".claude" / "skills" / "post-training-eval" / "SKILL.md").read_text()
        metadata = (CANONICAL / "agents" / "openai.yaml").read_text()

        self.assertTrue(skill.startswith("---\nname: post-training-eval\ndescription:"))
        self.assertIn("Never describe a diagnostic run", skill)
        self.assertIn("${CLAUDE_PROJECT_DIR}/.agents/skills/post-training-eval/SKILL.md", wrapper)
        self.assertIn("$post-training-eval", metadata)

    def test_installer_is_idempotent_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "install_skill.py"),
                "--agent", "both",
                "--mode", "copy",
                "--codex-root", str(base / "codex"),
                "--claude-root", str(base / "claude"),
            ]
            first = subprocess.run(command, check=False, capture_output=True, text=True)
            second = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)

            installed = base / "codex" / "post-training-eval" / "SKILL.md"
            installed.write_text("unrelated skill\n")
            refusal = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(refusal.returncode, 2)
            self.assertIn("refusing to overwrite", refusal.stdout)

    def test_skill_page_links_to_canonical_source(self) -> None:
        page = (ROOT / "docs" / "skill.html").read_text()
        self.assertIn("raw.githubusercontent.com/BirgerMoell/post-training-eval/main/.agents/skills/post-training-eval/SKILL.md", page)
        self.assertIn("scripts/install_skill.py --agent both", page)


if __name__ == "__main__":
    unittest.main()
