---
name: major-incident
description: How to run a major incident (several tickets or services failing at once from one cause) - grouping, parallel investigation, incident notes, fix, and close-out. Use when an outage spans multiple tickets or services.
---
# Major incident

1. **Scope**: list open tickets and triage them with `triage_tickets` (one call). Group the ones that share the
   failing services or started at the same time. Leave unrelated tickets out of the incident.
2. **Parent ticket**: create one parent ticket for the incident and link the grouped tickets to it.
3. **Investigate in parallel** (two `task` calls in one step): `it_diagnostics` follows the dependency chain
   and logs; `change_analyst` correlates recent changes with when the symptoms started.
4. **Incident notes** `/incident/notes.md`: hypotheses with status (open / confirmed / refuted) and evidence,
   plus a short timeline. Update them when evidence arrives. Subagents can read them.
5. **Fix** the root cause (see root-cause-analysis). If the verification fails, refute that hypothesis in the
   notes, revise the todo list and continue.
6. **People**: page the team that owns the root-cause service (sev1 if customers or revenue are affected),
   or when a saved lesson says a human must act.
7. **Close out** after a PASSED verification: resolve the parent and all linked tickets in ONE step, record
   the incident in memory, and write a stakeholder update with the outage-communication skill.
