"""Level 1 - skills are discoverable and loadable."""

from agent.skills import load_all, load_skill, skills_index


def test_all_skills_have_name_and_description() -> None:
    skills = load_all()
    assert set(skills) == {"incident-triage", "root-cause-analysis", "outage-communication", "major-incident"}
    assert all(s["description"] and s["body"] for s in skills.values())


def test_index_is_short_and_load_returns_full_body() -> None:
    index = skills_index()
    assert "root-cause-analysis:" in index and "Dependencies" not in index  # only descriptions in the prompt
    assert "Dependencies" in load_skill.invoke({"name": "root-cause-analysis"})
    assert "Unknown skill" in load_skill.invoke({"name": "nope"})
