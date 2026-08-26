#!/usr/bin/env python3
"""Install the repository's canonical Agent Skill without overwriting existing data."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SKILL_NAME = "post-training-eval"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / ".agents" / "skills" / SKILL_NAME


def default_codex_root() -> Path:
    return Path.home() / ".agents" / "skills"


def default_claude_root() -> Path:
    return Path.home() / ".claude" / "skills"


def install(source: Path, destination: Path, mode: str, dry_run: bool) -> str:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return f"already installed: {destination} -> {source}"
        raise FileExistsError(f"refusing to replace existing symlink: {destination}")
    if destination.exists():
        source_skill = (source / "SKILL.md").read_bytes()
        installed_skill = destination / "SKILL.md"
        if mode == "copy" and installed_skill.is_file() and installed_skill.read_bytes() == source_skill:
            return f"already installed copy: {destination}"
        raise FileExistsError(f"refusing to overwrite existing path: {destination}")

    if dry_run:
        return f"would install ({mode}): {source} -> {destination}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source, target_is_directory=True)
    else:
        shutil.copytree(source, destination)
    return f"installed ({mode}): {destination}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install the post-training-eval skill for Codex and/or Claude Code")
    parser.add_argument("--agent", choices=("codex", "claude", "both"), default="both")
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--codex-root", type=Path, default=default_codex_root())
    parser.add_argument("--claude-root", type=Path, default=default_claude_root())
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not (SOURCE / "SKILL.md").is_file():
        raise SystemExit(f"canonical skill is missing: {SOURCE}")

    targets: list[tuple[str, Path]] = []
    if args.agent in ("codex", "both"):
        targets.append(("Codex", args.codex_root.expanduser() / SKILL_NAME))
    if args.agent in ("claude", "both"):
        targets.append(("Claude Code", args.claude_root.expanduser() / SKILL_NAME))

    try:
        for agent, destination in targets:
            print(f"{agent}: {install(SOURCE, destination, args.mode, args.dry_run)}")
    except (FileExistsError, OSError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
