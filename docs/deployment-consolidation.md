# ECS deployment-definition migration

No production resources are changed by this repository update. The release
renderer uses these three canonical templates:

- `ecs-scrutiny-api-task.json` for the HTTP API
- `worker-task-definition.json` for ingestion workers
- `ecs-scrutiny-worker-task.json` for scrutiny workers

Read-only AWS inspection on 2026-10-24 found all five definitions still attached
to active services in `jubeex-ai-cluster`:

- `jubeex-scrutiny-api` -> `jubeex-scrutiny-api:39` (1/1 running)
- `jubeex-scrutiny-worker` -> `jubeex-scrutiny-worker:138` (2/2 running)
- `jubeex-ingestion-worker` -> `jubeex-ingestion-worker:35` (2/2 running)
- `jubeex-scrutiny-agent-service` -> `jubeex-scrutiny-agent:88` (1/1 running)
- `jubeex-ai-api` -> `jubeex-ai-api:2` (desired 1, running 0)

Therefore `agent-task.json` and `ecs-ai-api-task.json` are not removable yet.
The legacy services also use port 4501 and mutable image tags, while the current
API uses port 8000 and an immutable digest; they are not interchangeable without
an explicit traffic and compatibility cutover.

Before migration, list clusters and services, describe each service, and record
its task-definition ARN and revision. Compare container command, port mappings,
health checks, CPU/memory, IAM roles, environment, and Secrets Manager references
against the three rendered definitions. Render into a new directory with
`scripts/prepare_ecs_release.py` and immutable image digests. Register new task
definition revisions, update one service at a time, wait for stability, verify
health and queue processing, and retain the previous revision for rollback.

Production prerequisites now include `JUBEEX_API_KEY`,
`JUBEEX_REMOTE_FILE_HOSTS`, and an HTTPS `JUBEEX_CALLBACK_URL`. The checked-in
templates still contain the existing callback endpoint; replace it through the
release renderer with the approved HTTPS endpoint. Do not register a rendered
revision until those values and the workspace S3 host have been reviewed.
