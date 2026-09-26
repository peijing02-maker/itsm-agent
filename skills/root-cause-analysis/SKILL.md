---
name: root-cause-analysis
description: Step-by-step method to find and fix the root cause of a failing service, including dependency checks, recent changes and verification. Use before fixing anything.
---
# Root-cause analysis

1. **Symptom**: read the ticket and check the affected service's health.
2. **Dependencies**: a service is often unhealthy because of something it depends on.
   Follow `depends_on` down to the deepest unhealthy service. That is where the problem is, not the symptom.
   Its logs usually say who or what causes it (e.g. which client holds the connections).
3. **What changed?** Compare when the symptoms started (metric history) with recent deploys and config
   changes (change_analyst). A change shortly before the start, on a service in the failure chain, is the
   prime suspect. Run this in parallel with step 2 when several services fail.
4. **Is it us or them?** For internet-facing problems, check the real website, DNS, TLS certificate,
   and the vendor's public status page (internet_checker subagent).
5. **Known fixes**: search the knowledge base and past incidents for the symptom.
6. **Fix** the cause with the least disruptive action: roll back a bad change rather than restart what it
   breaks; flush a cache rather than restart it. Give a clear reason. It needs human approval.
7. **Verify**: happens automatically after the fix (the service and its dependents are watched for a few
   minutes). If it FAILS, the fix only relieved a symptom: revise the plan (write_todos) and go back to step 2-3.
   Only after it PASSES, resolve the ticket(s) with a note that covers symptom, root cause, action, and evidence
   (before and after metrics).
