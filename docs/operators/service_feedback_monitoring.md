# Post-Simplification Monitoring

With the placeholder services removed, teams should monitor early adopter
feedback to ensure optional adapter guidance covers real-world needs.

## Feedback Channels
- **Issue tracker**: label requests related to search or caching with
  `needs-adapter` to identify trends quickly.
- **Support slack**: pin a standing reminder to reference the adapter guide so
  requests gather context on existing recipes.

## Weekly Review Checklist
1. Audit new tickets tagged `needs-adapter`.
2. Confirm whether existing recipes address the request; if not, record the gap
   in the technical debt log.
3. Track the number of open adapter-related issues. Escalate if the count grows
   for two consecutive weeks.

## Escalation Criteria
- Three or more production incidents tied to missing adapters within a month.
- A single customer request that requires repository changes beyond documented
  recipes.

Use this checklist during release triage to decide whether to invest in new
adapters or additional automation.
