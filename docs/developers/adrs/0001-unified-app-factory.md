# ADR 0001: Unify FastAPI App Factory with Profile-Driven Composition

**Date:** 2025-02-14  
**Status:** Accepted  
**Drivers:** Dual mode architecture drift, broken service resolution, duplicated CORS/middleware logic, security gaps on config endpoints  
**Deciders:** AI Docs Platform Team

## Context

- The current `create_app` implementation bifurcates simple vs enterprise modes, but the enterprise stack is unfinished and the simple stack calls through a global service factory singleton that never receives the per-app registrations. As a result the primary `/search` endpoint fails in practice.
- `ModeConfig.enabled_services` enumerates conceptual names (`"basic_search"`, `"qdrant_client"`) that do not align with the service registry keys (`"vector_db_service"`, `"search_service"`, `"cache_service"`). The gate in `ModeAwareServiceFactory.get_service` therefore blocks real services.
- Two competing CORS implementations exist (one in `app_factory`, one in the middleware manager), and configuration routes expose reload/rollback features without authentication, violating minimum security expectations.
- Maintaining two factories doubles wiring effort, encourages dead code, and violates KISS/DRY/YAGNI goals without delivering working differentiation.

## Decision

- Collapse the dual factory into a single profile-driven `create_app(profile)` that composes feature installers based on configuration and documented presets (simple, enterprise).
- Bind the request-aware `ModeAwareServiceFactory` to FastAPI's dependency system and deprecate reliance on the global singleton helper. Routers retrieve services through request-scoped dependencies, ensuring registrations performed during startup are visible at request time.
- Normalize `ModeConfig.enabled_services` and middleware stacks to canonical identifiers, pruning aspirational entries until implementations land. Treat CORS as an explicit concern owned by the app factory and keep middleware manager surface limited to supported keys.
- Guard configuration management routes with an API-key dependency that fails closed when `config.security.api_key_required` is enabled, acknowledging a follow-up task to integrate with a secure key store.

## Consequences

- Shared infrastructure (logging, vector DB, embeddings) is wired once, reducing cognitive load and drift between profiles.
- Tests and CI can execute against a deterministic matrix of supported profiles instead of ad-hoc mode splits.
- Existing callers of the legacy global service helpers continue to work during transition because the per-app factory is registered with the singleton, but new work should prefer explicit dependencies.
- Security posture improves: CORS policy is applied exactly once with predictable headers, and high-impact config APIs require authorization by default.
- Upcoming enterprise features must be added as explicit installers with lazy imports and opt-in configuration flags. This keeps the codebase aligned with YAGNI while leaving an extensibility path.

## Status Notes

Follow-up tasks include replacing placeholder API-key verification with the real key management service, migrating remaining routers off the singleton helper, and expanding the installer pattern to cover analytics/observability when those modules are production ready.
