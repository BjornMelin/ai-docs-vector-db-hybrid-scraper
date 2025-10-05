# Release Notes – Configuration & Security Simplification

## Highlights

- Configuration reload API now validates source paths and surfaces user errors as HTTP 400 responses.
- File watching requires an explicit configuration source and verifies osquery readiness before enabling.
- Security middleware trims unused configuration fields and now has dedicated header + rate limiting tests.
- Legacy enterprise service scaffolding has been removed; optional services fail closed until replacements land.

## Operator Actions

- Ensure `OSQUERY_RESULTS_LOG` points at a live `file_events` stream before enabling `/config/file-watch/enable`.
- Update deployment templates to use the renamed `security.api_key_required` and `security.api_keys` fields.
- Factor the removal of enterprise search/cache modules into any custom integrations that imported them.
