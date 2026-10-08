# Performance deployment (ECS)

These commands deploy one image into two isolated worker services:

- `jubeex-ingestion-worker`: Split + Extraction, 2 vCPU / 4 GiB, OCR 4.
- `jubeex-scrutiny-worker`: Scrutiny only, 1 vCPU / 2 GiB.

The scrutiny task starts through `scripts/start.sh`. The script reads the
external-account WIF configuration from AWS Secrets Manager, exposes the ECS
task-role credentials, and Google ADC exchanges them for short-lived Vertex
credentials. AWS hosts the worker; Vertex AI remains the model provider.

The API service does not need to execute long jobs. It continues to enqueue
into `jubeex-ingestion-jobs` and `jubeex-scrutiny-jobs`.

## 1. Validate locally

```bash
cd /Users/admin/Documents/codes/Jubeex-Scrutiny-Agent
uv sync --locked
uv run pytest -q \
  tests/test_llm.py \
  tests/test_scrutiny_collect.py \
  tests/test_scrutiny_e2e.py \
  tests/test_workflow.py \
  tests/test_structure_split.py \
  tests/test_split_ocr.py \
  tests/test_worker.py \
  tests/test_queue.py
```

## 2. Build and push one immutable worker image

```bash
export AWS_REGION="ap-south-1"
export AWS_ACCOUNT_ID="893338224943"
export ECS_CLUSTER="jubeex-ai-cluster"
export ECR_REPOSITORY="jubeex-scrutiny-agent"
export IMAGE_TAG="performance-$(git rev-parse --short HEAD)-$(date +%Y%m%d%H%M%S)"
export IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG"

aws sts get-caller-identity
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin \
  "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.worker \
  -t "$IMAGE_URI" \
  --push .
```

## 3. Register both task definitions with that image

Do not edit task-definition image tags by hand. Render temporary registration
files so both services run the identical build.

```bash
jq --arg image "$IMAGE_URI" \
  '.containerDefinitions[0].image = $image' \
  worker-task-definition.json > /tmp/jubeex-ingestion-worker.json

jq --arg image "$IMAGE_URI" \
  '.containerDefinitions[0].image = $image' \
  ecs-scrutiny-worker-task.json > /tmp/jubeex-scrutiny-worker.json

export INGESTION_TASK_ARN="$(aws ecs register-task-definition \
  --cli-input-json file:///tmp/jubeex-ingestion-worker.json \
  --region "$AWS_REGION" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)"

export SCRUTINY_TASK_ARN="$(aws ecs register-task-definition \
  --cli-input-json file:///tmp/jubeex-scrutiny-worker.json \
  --region "$AWS_REGION" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)"

echo "$INGESTION_TASK_ARN"
echo "$SCRUTINY_TASK_ARN"
```

## 4. Create the ingestion service, then narrow the existing worker

Create ingestion first. This avoids a period where no task consumes ingestion
messages. It reuses the existing worker service's VPC configuration.

```bash
export NETWORK_CONFIGURATION="$(aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services jubeex-scrutiny-worker \
  --region "$AWS_REGION" \
  --query 'services[0].networkConfiguration' \
  --output json)"

aws ecs create-service \
  --cluster "$ECS_CLUSTER" \
  --service-name jubeex-ingestion-worker \
  --task-definition "$INGESTION_TASK_ARN" \
  --desired-count 4 \
  --launch-type FARGATE \
  --network-configuration "$NETWORK_CONFIGURATION" \
  --region "$AWS_REGION"

aws ecs wait services-stable \
  --cluster "$ECS_CLUSTER" \
  --services jubeex-ingestion-worker \
  --region "$AWS_REGION"

aws ecs update-service \
  --cluster "$ECS_CLUSTER" \
  --service jubeex-scrutiny-worker \
  --task-definition "$SCRUTINY_TASK_ARN" \
  --desired-count 4 \
  --force-new-deployment \
  --region "$AWS_REGION"

aws ecs wait services-stable \
  --cluster "$ECS_CLUSTER" \
  --services jubeex-scrutiny-worker \
  --region "$AWS_REGION"
```

For later deployments, update both existing services instead of running
`create-service`:

```bash
aws ecs update-service --cluster "$ECS_CLUSTER" \
  --service jubeex-ingestion-worker --task-definition "$INGESTION_TASK_ARN" \
  --force-new-deployment --region "$AWS_REGION"
aws ecs update-service --cluster "$ECS_CLUSTER" \
  --service jubeex-scrutiny-worker --task-definition "$SCRUTINY_TASK_ARN" \
  --force-new-deployment --region "$AWS_REGION"
```

## 5. Add a safe autoscaling baseline

```bash
for service in jubeex-ingestion-worker jubeex-scrutiny-worker; do
  aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --resource-id "service/$ECS_CLUSTER/$service" \
    --scalable-dimension ecs:service:DesiredCount \
    --min-capacity 2 \
    --max-capacity 20 \
    --region "$AWS_REGION"

  aws application-autoscaling put-scaling-policy \
    --service-namespace ecs \
    --resource-id "service/$ECS_CLUSTER/$service" \
    --scalable-dimension ecs:service:DesiredCount \
    --policy-name "$service-cpu-60" \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration \
      '{"TargetValue":60,"PredefinedMetricSpecification":{"PredefinedMetricType":"ECSServiceAverageCPUUtilization"},"ScaleOutCooldown":60,"ScaleInCooldown":300}' \
    --region "$AWS_REGION"
done
```

CPU scaling is a safety baseline. For production load, add SQS queue-depth
alarms because parse and LLM tasks can wait on remote services without using
much CPU.

## 6. Verify the rollout and timings

```bash
aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services jubeex-ingestion-worker jubeex-scrutiny-worker \
  --region "$AWS_REGION" \
  --query 'services[].{service:serviceName,desired:desiredCount,running:runningCount,pending:pendingCount,task:taskDefinition,rollout:deployments[0].rolloutState}' \
  --output table

aws logs tail /ecs/jubeex-scrutiny-agent \
  --since 30m --follow --region "$AWS_REGION" | \
  grep -E 'SplitTiming|ExtractionTiming|ScrutinyTiming|LLM provider'
```

Expected configuration:

```text
ingestion: JUBEEX_WORKER_KINDS=process_file, SPLIT_OCR_CONCURRENCY=4
scrutiny:  JUBEEX_WORKER_KINDS=scrutiny, LLM_PROVIDER=vertex
```

The scrutiny log must also contain these startup lines before the worker begins:

```text
[startup] WIF configuration created
[startup] Google ADC/WIF environment prepared
```

If Vertex returns quota errors, set `VERTEX_REQUESTS_PER_MINUTE` to the
project's actual quota instead of restoring the OpenRouter-only limit.
