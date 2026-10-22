# ECS worker release and recovery checklist

This repository contains task definitions and worker code, not the live ECS
service, SQS, or Application Auto Scaling configuration. Applying this code
alone does **not** change production. Do not deploy the changes below until
the operator has reviewed queue and IAM settings and run the acceptance tests.

## Verify the service that actually runs scrutiny

The legacy `jubeex-scrutiny-agent-service` uses the
`jubeex-scrutiny-agent` task-definition family. The dedicated SQS consumer in
this repository uses the `jubeex-scrutiny-worker` family, runs
`jubeex-sqs-worker`, and sets `JUBEEX_WORKER_KINDS=scrutiny`. Updating the legacy
service does **not** prove that the SQS scrutiny consumer was updated. In
particular, registering a task revision does not deploy it, and seeing a new
service task-definition ARN does not by itself prove that every running task
uses that revision or a new image.

There is a second route to Vertex: the backend's current Run Scrutiny call
posts to `/v1/filings/{id}/index`. That endpoint enqueues an `index_parsed`
`process_file` job, which first indexes Pinecone vectors and then runs defects
inside the ingestion worker. The defects-only `/scrutiny` endpoint instead
enqueues on the dedicated scrutiny queue. Therefore both worker services need
the same shared Vertex pacing configuration and S3 access; checking only the
dedicated scrutiny worker is insufficient for the main UI flow.

Run these read-only checks in the AWS account that hosts production:

```sh
aws ecs describe-services --cluster jubeex-ai-cluster \
  --services jubeex-scrutiny-agent-service jubeex-ingestion-worker jubeex-scrutiny-worker \
  --region ap-south-1 \
  --query 'services[].{service:serviceName,task:taskDefinition,desired:desiredCount,running:runningCount,deployments:deployments[].{task:taskDefinition,state:rolloutState,running:runningCount}}' \
  --output json

for family in jubeex-ingestion-worker jubeex-scrutiny-worker; do
  aws ecs describe-task-definition --task-definition "$family" \
    --region ap-south-1 \
    --query 'taskDefinition.{revision:revision,containers:containerDefinitions[].{name:name,image:image,command:command,environment:environment[?name==`JUBEEX_WORKER_KINDS` || name==`LLM_PROVIDER` || name==`VERTEX_SHARED_RATE_LIMIT` || name==`VERTEX_REQUESTS_PER_MINUTE` || name==`LLM_REQUESTS_PER_MINUTE` || name==`AWS_S3_BUCKET`]}}' \
    --output json
done
```

The loop inspects the latest registered revision of each worker; compare each
ARN with that worker service's active task definition before drawing conclusions.
Also inspect running tasks with `aws ecs list-tasks` and `aws ecs describe-tasks`
to confirm their `taskDefinitionArn` and container `imageDigest`. Verify the
worker task role can read and conditionally write the rate-limit object in the
artifact S3 bucket. With `VERTEX_SHARED_RATE_LIMIT=true`, an absent bucket or
failed S3 access causes requests to fail closed. `SCRUTINY_CONCURRENCY=8`
controls parallel checks inside a job; it does not mean eight Vertex requests
per minute or eight jobs. Even correctly shared pacing cannot guarantee the
absence of Vertex `429 RESOURCE_EXHAUSTED` responses.

## Required AWS configuration

1. Keep separate `jubeex-ingestion-jobs` and `jubeex-scrutiny-jobs` queues and
   separate ECS services. Configure each source queue with a dead-letter queue
   and a bounded `RedrivePolicy` (for example, `maxReceiveCount=5`). Alarm on
   DLQ depth, oldest message age, worker task restarts, and missing completions.
   The application no longer auto-creates queues without a DLQ.
2. Confirm both worker task roles have least-privilege S3 read/write access to
   `agent-jobs/*` in the artifact bucket. The job status and conditional lease
   require `GetObject` and `PutObject`. Keep SQS receive/delete/change-visibility
   permissions on the matching queue only. Check Secrets Manager access for
   `jubeex/google-wif-credentials`.
3. Scale each ECS service from its own SQS backlog per running task, and set a
   bounded minimum/maximum. Use queue wait-time and real job-duration targets
   to choose the scaling target. Scale-out is not a remedy for Vertex 429s:
   measure model request rate, in-flight calls, token volume, 429s, and latency
   before increasing scrutiny task count. Protect active long-running tasks
   from scale-in where possible.
4. Build API, ingestion worker, and scrutiny worker from the same reviewed
   commit and deploy immutable ECR digests. The API and worker use different
   Dockerfiles, so build and pin one digest for each image. Current checked-in
   task definitions contain historical tags; verify actual ECS revisions and
   image digests before rollout. Use a canary rather than changing all services
   at once.
   After building, render immutable task definitions locally:

   ```sh
   python scripts/prepare_ecs_release.py \
     --api-image REPOSITORY@sha256:DIGEST \
     --worker-image REPOSITORY@sha256:DIGEST \
     --callback-url https://BACKEND/webhook \
     --output-dir /private/tmp/jubeex-ecs-release
   ```

   The command validates the full DB catalogue, required secrets and dedicated
   worker roles; it does not register tasks or deploy services. Replace the
   placeholders with actual digest references and the approved backend webhook
   endpoint.
5. Configure S3 lifecycle expiry for old `agent-jobs/*/lease.json` objects and
   job-status retention according to the application's audit policy.
6. The checked-in ECS and LlamaDeploy templates now use
   `SCRUTINY_CATALOGUE_SOURCE=db` and `SCRUTINY_DEFECTS=all`. This is a release
   default, not a live change. Confirm the catalogue reader secret exists, the
   enabled production defect count is expected, and the API and worker log the
   same catalogue version. The 11-code JSON allowlist is only a deliberate
   test override; never carry it into a full-catalogue production rollout.
7. Verify callback transport and backend reachability. Legacy task definitions
   include an HTTP callback URL; replace it with an approved private endpoint
   or HTTPS URL before registration. Do not assume this repo provisions DNS,
   TLS, security groups, or the backend webhook receiver.
   When `JUBEEX_CALLBACK_URL` is set, request-supplied callback URLs no longer
   override it by default. `JUBEEX_CALLBACK_ALLOW_REQUEST_OVERRIDE=true` is an
   explicit integration-test escape hatch, not a production setting.
8. Set `JUBEEX_WORKER_MAX_JOB_SECONDS` to a measured upper bound for a filing
   (default 5400 seconds). A workflow exceeding this deadline becomes a failed
   job with a callback; it must not remain at 90% indefinitely. Alert on jobs
   approaching the deadline, and investigate slow Vertex calls instead of
   simply raising it repeatedly.

## Pre-deployment acceptance tests

- Run one upload through split/extraction and one scrutiny through Review.
- Start two scrutiny jobs simultaneously. Both must finish without duplicate
  callbacks, duplicated defects, or a stuck progress display.
- Force one Vertex 429. The job may retry, but if a check remains failed, the
  overall job must be `failed`, not `completed`; partial findings remain saved.
- Kill a worker during a job. SQS must redeliver, and the S3 lease must prevent
  concurrent execution of that job. On graceful SIGTERM, the worker cancels
  the job and makes its SQS message visible again. Verify the eventual terminal
  state; a hard kill may still wait for the S3 lease to expire.
- Make the backend callback return an error. The terminal SQS message must
  retry callback delivery with the same event ID; the backend deduplicates it.
- Check credentials after an ECS task has run longer than the task-role
  credential lifetime. AWS SDK calls and Google WIF must still work.
- Confirm the `gemini-4-argon` model is enabled for the project at `global` and
  measure cost and accuracy against the 3.8 Flash baseline. Argon has different
  latency and pricing; switching models is not itself an accuracy guarantee.

No queue purge, ECS update, image push, or production configuration change is
performed by this checklist.
