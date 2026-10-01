# ECS worker scaling

The SQS worker intentionally processes one petition at a time. Parallelism is
provided by ECS tasks so OCR-heavy petitions do not compete inside one Python
process.

## Immediate ingestion scaling

Set the existing ingestion worker task definition environment variable:

```text
JUBEEX_WORKER_KIND=ingestion
```

For the sub-two-minute target on typical 100-page scanned petitions, use 4 vCPU
and 8 GB memory per ingestion task and set:

```text
SPLIT_OCR_WORKERS=4
SPLIT_OCR_DPI=180
```

The OCR worker count should match the task vCPU count. Deploy the queue-based
autoscaling stack with the actual cluster and service names:

### Replace the legacy 0/1 policies first

If the service already has `jubeex-worker-scale-out` and
`jubeex-worker-scale-in`, remove those policies and their alarms before
deploying this stack. The legacy policies use `ExactCapacity` 1/0 and cap the
scalable target at one task, so leaving them active would fight the new 1-4
task policy.

```bash
aws application-autoscaling delete-scaling-policy \
  --region ap-south-1 \
  --service-namespace ecs \
  --resource-id service/jubeex-ai-cluster/jubeex-scrutiny-worker \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name jubeex-worker-scale-out

aws application-autoscaling delete-scaling-policy \
  --region ap-south-1 \
  --service-namespace ecs \
  --resource-id service/jubeex-ai-cluster/jubeex-scrutiny-worker \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name jubeex-worker-scale-in

aws cloudwatch delete-alarms \
  --region ap-south-1 \
  --alarm-names \
    jubeex-worker-ingestion-scale-out \
    jubeex-worker-scrutiny-scale-out \
    jubeex-worker-scale-in
```

These commands remove only the obsolete scaling controls; they do not stop
the running ECS task or delete either SQS queue.

### Deploy the replacement policy

```bash
aws cloudformation deploy \
  --region ap-south-1 \
  --stack-name jubeex-ingestion-worker-scaling \
  --template-file infra/ecs-ingestion-worker-autoscaling.yaml \
  --parameter-overrides \
    EcsClusterName=jubeex-ai-cluster \
    EcsServiceName=jubeex-scrutiny-worker \
    IngestionQueueName=jubeex-ingestion-jobs \
    MinTasks=1 \
    MaxTasks=4
```

The service scales out when ingestion messages are visible. It scales in only
after both queued and in-flight messages remain at zero for fifteen minutes,
so a long-running split is not terminated by an empty visible queue.

## Separate scrutiny service

Create a second ECS service from the same image/task role and set:

```text
JUBEEX_WORKER_KIND=scrutiny
```

Give the ingestion service `JUBEEX_WORKER_KIND=ingestion`. Both services may
retain both queue URLs because the worker-kind setting determines which queue
each service consumes. Start the scrutiny service with one task and scale it
independently if scrutiny backlog grows.

Do not set in-process ingestion concurrency above one. Increasing the ECS task
count is the isolation boundary for simultaneous PDF rendering and Tesseract
work.
