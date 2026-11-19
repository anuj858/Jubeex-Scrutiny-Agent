# Jubeex scrutiny environment infrastructure

The existing `jubeex-ai-cluster` is the combined `dev-staging` environment.
`production-agent-base.yaml` creates isolated production ECR, ECS, SQS,
CloudWatch, IAM, and private service-discovery resources. It imports only the
production artifact bucket created by the full-application core stack.
`production-services.yaml` bootstraps the four ECS services after their first
production task definitions are registered. Both worker services start at one
task and scale between one and four tasks at a 60% CPU target.

## CI/CD behavior

`.github/workflows/deploy.yml` performs tests first, builds three immutable
images (API, worker, and agent), registers environment-specific task-definition
revisions, and updates ECS with the deployment circuit breaker enabled.

- Push to `main`: deploys automatically to `production` after quality checks.
- Push to `jubeex-dev`: deploys automatically to `dev-staging` after quality checks.
- Manual reruns use the selected branch, so `main` remains production-only and
  `jubeex-dev` remains dev/staging-only.
- AWS authentication uses environment-scoped GitHub OIDC roles.
- Dev/staging and production use separate buckets, queues, secrets, cluster,
  task roles, log group, and Pinecone namespaces.

The GitHub `production` environment should require approval before a workflow
can assume the production AWS role.

## Data-residency boundary

All resources created by this template are in AWS Mumbai (`ap-south-1`). The
current external AI path is not automatically India-only. OpenRouter,
LlamaCloud, Pinecone, and any upstream model provider must be contractually and
technically validated before production legal documents are submitted.
