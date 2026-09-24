"""Skills: reusable know-how in markdown files (skills/<name>/SKILL.md).

Progressive disclosure: only each skill's name and description go into the system
prompt (cheap). The agent calls `load_skill` to read the full instructions when a
task needs them. Adding a skill only requires a new folder; no code changes.
"""

from pathlib import Path

from langchain_core.tools import tool

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _parse(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    _, front, body = text.split("---", 2)
    meta = dict(line.split(":", 1) for line in front.strip().splitlines())
    return {"name": meta["name"].strip(), "description": meta["description"].strip(), "body": body.strip()}


def load_all(skills_dir: Path = SKILLS_DIR) -> dict[str, dict[str, str]]:
    skills = (_parse(p) for p in sorted(skills_dir.glob("*/SKILL.md")))
    return {s["name"]: s for s in skills}


def skills_index() -> str:
    """One line per skill, for the system prompt."""
    return "\n".join(f"- {s['name']}: {s['description']}" for s in load_all().values())


@tool
def load_skill(name: str) -> str:
    """Load the full instructions of a skill by name (see the skills list in your instructions)."""
    skills = load_all()
    if name not in skills:
        return f"Unknown skill '{name}'. Available: {sorted(skills)}"
    return skills[name]["body"]
