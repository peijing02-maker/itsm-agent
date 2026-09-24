---
name: root-cause-analysis
description: Step-by-step method to find and fix the root cause of a failing service, including dependency checks and verification. Use before restarting anything.
---
# Root-cause analysis

1. **Symptom**: read the ticket and check the affected service's health.
2. **Dependencies**: a service is often unhealthy because of something it depends on.
   Check every service in `depends_on`. The root cause is the unhealthy dependency, not the symptom.
3. **Is it us or them?** For internet-facing problems, check the real website, DNS, TLS certificate,
   and the vendor's public status page (internet_checker subagent).
4. **Known fixes**: search the knowledge base for the symptom.
5. **Fix**: restart ONLY the root-cause service, giving a clear reason. This needs human approval.
6. **Verify**: re-check the originally affected service. Only then resolve the ticket, with a note that
   covers symptom, root cause, action, and evidence (before and after metrics).
