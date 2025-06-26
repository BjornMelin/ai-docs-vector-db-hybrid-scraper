# Configuration Reload Example

This project includes a simple configuration reload mechanism for development convenience.

## Features

- **Automatic Reload**: In development mode, the application automatically watches for changes to `.env` files and reloads configuration
- **Manual Reload**: A manual reload endpoint is available at `POST /api/config/reload` (development mode only)
- **Debouncing**: Changes are debounced to prevent multiple rapid reloads
- **Simple Implementation**: Under 100 lines of code with no complex features

## Usage

### Automatic Reload (Development Mode)

When running in development mode (`AI_DOCS_ENVIRONMENT=development`), the config watcher starts automatically:

```bash
# Start the API server
uv run uvicorn src.api.main:app --reload

# The watcher will log when it starts:
# "Configuration watcher started in development mode"
```

Now when you modify the `.env` file, the configuration will automatically reload after a 1-second debounce period.

### Manual Reload

You can also trigger a manual reload via the API:

```bash
curl -X POST http://localhost:8000/api/config/reload
```

Response:
```json
{
  "status": "success",
  "message": "Configuration reloaded",
  "environment": "development"
}
```

## Configuration

The watcher only runs in development mode. To enable it:

```env
AI_DOCS_ENVIRONMENT=development
```

## Testing

To test the configuration reload:

1. Start the API server in development mode
2. Check the current configuration:
   ```bash
   curl http://localhost:8000/info
   ```
3. Modify the `.env` file (e.g., change `AI_DOCS_DEBUG=true`)
4. Wait 1-2 seconds for the automatic reload
5. The logs will show: "Configuration reloaded successfully"

## Implementation Details

- Uses the `watchdog` library to monitor file system changes
- Only watches for `.env` file changes (including `.env.local`, `.env.development`, etc.)
- Implements debouncing to prevent multiple rapid reloads
- Integrates with FastAPI's lifespan management for proper startup/shutdown
- No complex features like rollback, drift detection, or observability beyond basic logging