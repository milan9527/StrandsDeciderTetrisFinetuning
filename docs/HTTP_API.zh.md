# Decider 微调 HTTP 服务

HTTP 服务将请求放入持久队列，GPU worker 依次调用与 `finetune.py` 相同的入口。
数据审计、SFT、评估、校准、重新加载验证和质量门槛继续由原流程负责。
API 进程重启不会停止独立 worker 中的训练。

当前版本面向**一个受信任团队、单台 Linux GPU 主机、一个 worker**。
API 与 worker 共享本地磁盘上的 SQLite 和任务目录；不支持多主机共享数据库、
不同租户的数据隔离或自动扩缩容。所有持有同一 token 的调用者可以查看全部任务。
模型、解释器、上游代码位置由管理员配置，请求不能覆盖这些执行设置。

## 1. 在已有 GPU EC2 直接运行

已验证的训练规格为 `g6.xlarge` / L4 24GB，完整训练预检要求至少 18GiB 空闲显存。
主机需要兼容 CUDA 12.6 的驱动、Python 3.11+、Git 和 uv。
仓库为私有时，使用者需要 GitHub 读取权限；模型权重另行准备。

```bash
uv sync --locked --extra api
```

编辑 `examples/scenario-finetune.json` 的两个模型目录：

```json
{
  "checkpoint": "/opt/models/decider",
  "base_model": "/opt/models/base"
}
```

模型要求及数据格式见 [统一入口说明](FINETUNE.md)。`auto_setup: true` 会在首次训练时安装固定版本
Python 依赖，不会安装主机驱动。也可以由管理员提供已准备的训练环境。

编辑 `examples/http-service.json`：

```json
{
  "storage_dir": "/srv/decider/storage",
  "pipeline_config": "/path/to/scenario-finetune.json",
  "local_data_roots": ["/srv/decider/data"],
  "s3_data_prefixes": ["s3://YOUR_BUCKET/datasets/"],
  "max_pending_jobs": 100
}
```

配置文件内的相对路径以该配置文件所在目录为基准。`storage_dir` 需要可写，建议使用持久 EBS。
本地数据路径是**服务端路径**，API 和 worker 必须能在相同绝对路径读取它。
不使用的本地/S3 白名单可以设为 `[]`。不要把整个主机根目录加入数据白名单。

启动 API：

```bash
export DECIDER_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
# 将此 token 保存在自己的 secret 管理系统中，并提供给受信任的客户端。
uv run --extra api python -m finetuning.http_api \
  --config examples/http-service.json --host 127.0.0.1 --port 8080
```

在第二个终端、同一工作目录启动 worker：

```bash
uv run --extra api python -m finetuning.worker --config examples/http-service.json
```

worker 不需要 API token。每个存储目录只允许一个 worker；重复启动会拒绝运行。
同一块 GPU 应只分配给一个这样的队列。

API 默认只监听本机。跨机器访问时，将它置于带 HTTPS 的 ALB 或反向代理后，并将安全组限制到
调用方或负载均衡器；不要把 bearer token 放入 URL。
ALB 直接访问 EC2 时，原生启动需使用 `--host 0.0.0.0`；
Compose 的端口映射需改为 `8080:8080`，并只允许 ALB 安全组访问该端口。

## 2. 提交与查询

以下命令在已设置 `DECIDER_API_TOKEN` 的客户端执行，`jq` 用来提取任务 ID：

```bash
export DECIDER_API_URL=http://127.0.0.1:8080

JOB_ID=$(curl --fail-with-body -sS "$DECIDER_API_URL/v1/jobs" \
  -H "Authorization: Bearer $DECIDER_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: browser-v1-request-001" \
  -d '{
    "data": "s3://YOUR_BUCKET/datasets/browser/",
    "mode": "train",
    "overrides": {
      "training": {"epochs": 1, "eval_every": 50},
      "quality": {"min_accuracy": 0.8, "min_order_consistency": 0.9}
    }
  }' | jq -r .id)

curl --fail-with-body -sS "$DECIDER_API_URL/v1/jobs/$JOB_ID" \
  -H "Authorization: Bearer $DECIDER_API_TOKEN"
```

提交立即返回 HTTP 202 和任务 ID。相同 `Idempotency-Key` 与相同解析后配置返回原任务；
同一 key 改变请求返回 409。队列满返回 429。重新训练需要使用新的 key。

`mode` 支持：

| 值 | 实际工作 |
| --- | --- |
| `train` | 完整 GPU 微调与验证，默认值 |
| `validate` | 只运行实际数据导入和审计，无需 GPU 或权重，不证明训练成功 |
| `smoke` | 实际权重的一步训练及小样本验证，不证明微调效果 |

首次部署可以先用 `mode: "validate"` 检查数据与服务链路。`smoke` 的设备由管理员的
`runtime.device` 决定。HTTP 调用者不能切换设备或传入任意命令。

`overrides` 可使用统一配置中的 `training`、`evaluation`、`quality`，以及
`data.mode`、`data.fractions`、`data.seed`。不支持覆盖 `model`、`runtime`、
数据位置或输出目录；数据位置统一使用请求的 `data` 字段。

```bash
# 分页列出任务
curl -sS "$DECIDER_API_URL/v1/jobs?limit=20&offset=0" \
  -H "Authorization: Bearer $DECIDER_API_TOKEN"

# 读取增量日志；下次用返回的 next_offset，偏移量以字节计
curl -sS "$DECIDER_API_URL/v1/jobs/$JOB_ID/logs?offset=0&limit=65536" \
  -H "Authorization: Bearer $DECIDER_API_TOKEN"

# 训练阶段的详细日志
curl -sS "$DECIDER_API_URL/v1/jobs/$JOB_ID/logs?name=output/logs/train.log" \
  -H "Authorization: Bearer $DECIDER_API_TOKEN"

# 取消排队或运行中的任务；运行中先进入 cancelling
curl -sS -X POST "$DECIDER_API_URL/v1/jobs/$JOB_ID/cancel" \
  -H "Authorization: Bearer $DECIDER_API_TOKEN"

# 列出可下载产物，包括失败任务已经写入的日志和检查点
curl -sS "$DECIDER_API_URL/v1/jobs/$JOB_ID/artifacts" \
  -H "Authorization: Bearer $DECIDER_API_TOKEN"

# 下载最终报告；文件尚未生成时返回 404
curl --fail-with-body -sS \
  "$DECIDER_API_URL/v1/jobs/$JOB_ID/artifacts/output/report.json" \
  -H "Authorization: Bearer $DECIDER_API_TOKEN" -o report.json
```

`worker.log` 提供阶段进度和入口错误；`output/logs/*.log` 提供训练/评估详细输出。
日志单次最多读取 1MiB。检查点目录为 `output/eval-v1/adapted/calibrated/`，通过产物列表
逐文件下载；使用时仍需同一基座。结果保存在服务磁盘，**不会自动上传 S3 或部署推理服务**。

任务状态：

| 状态 | 含义 |
| --- | --- |
| `queued` | 等待 worker |
| `running` | 正在执行；`current_stage` 表示入口阶段 |
| `succeeded` | 所选 mode 的流程完成；只有 `mode=train` 才代表完整训练且质量达标 |
| `quality_failed` | 完整流程完成，质量门槛未通过，入口退出码 2；产物仍保留 |
| `failed` | 数据、环境、训练、重载或其他执行失败 |
| `cancelling` / `cancelled` | 已请求取消 / 已取消 |
| `interrupted` | worker 停机或上次异常退出，任务未获得完整结果 |

API 重启后可以继续查询任务。worker 正常停机将中断当前任务并清理训练进程组；
异常重启时先清理上次任务的遗留进程，再将任务标记为 `interrupted`。
排队任务继续执行，失败/中断任务不自动重试。当前 HTTP 版本不暴露 `--resume`；
管理员仍可在确认代码和输入不变后使用 CLI 的阶段恢复功能。

`GET /healthz` 无需认证，只检查 API 存活。`GET /readyz` 需要认证并检查 worker 心跳，
不代表 GPU 或模型预检已通过。`GET /openapi.json` 需要认证，提供完整接口 schema。
所有时间戳为 Unix 秒。

## 3. 用 Docker Compose 部署到自己的 AWS

仓库提供 API 与 GPU worker 两个构建目标，配置文件不含任何固定 AWS 账户或实例 ID：

- [Dockerfile](../deploy/http/Dockerfile)：API 轻量镜像，GPU 镜像预装固定训练依赖和上游代码。
- [compose.yaml](../deploy/http/compose.yaml)：单机 API、worker、持久存储和 GPU 映射。
- [服务配置](../deploy/http/config/service.json) / [训练配置](../deploy/http/config/pipeline.json)。

需要 Linux x86_64、Docker Compose v2、NVIDIA 驱动和 NVIDIA Container Toolkit。
GPU 镜像目标是 `linux/amd64`，不能在 ARM 主机上原生构建或训练；API 镜像也支持 ARM。
镜像不包含模型权重。
worker 镜像构建会下载 PyTorch CUDA 依赖，建议预留至少 30GB 构建空间，并给模型与任务留足 EBS。

```bash
sudo install -d -o 10001 -g 10001 \
  /srv/decider/storage /srv/decider/data /srv/decider/models /srv/decider/cache
# 将对应模型放入 /srv/decider/models/decider 和 /srv/decider/models/base，
# 确保 UID 10001 可以读取。数据放入 /srv/decider/data，或使用自己的 S3。
# 编辑 deploy/http/config/service.json 中的 S3 白名单。

export DECIDER_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export AWS_DEFAULT_REGION=us-east-1
docker compose -f deploy/http/compose.yaml up --build -d
```

可通过 `DECIDER_STORAGE_DIR`、`DECIDER_DATA_DIR`、`DECIDER_MODELS_DIR`、`DECIDER_CACHE_DIR`
替换宿主机目录。首次启动前创建目录并授予 UID 10001 所需权限；输入和模型在容器内只读挂载。
SQLite 与任务目录必须位于同一主机的持久磁盘，不能把此配置直接放到多主机 EFS 上扩容。

EC2 instance profile 至少需要对允许的数据前缀具有 `s3:GetObject`；读取分区目录还需要
`s3:ListBucket`。如果使用 CMK 加密，再授予对应 `kms:Decrypt`。
Docker bridge 模式读取 instance profile 时，管理员需要允许容器访问 IMDSv2，并按 AWS
容器访问要求设置实例 metadata response hop limit（通常为 2）。
不要将 AWS access key 或 API token 写入镜像、仓库和训练配置。

存储中会包含原始数据快照及模型产物，应按团队的数据保留策略备份和清理。可由管理员将已结束任务
同步到自己账户的 S3。通过 ALB 使用时，可以用 `/healthz` 检查 API，并单独监控 `/readyz` 和 worker 日志。

如果需要多台 GPU 的托管任务队列，可以继续增加 AWS Batch EC2 GPU 后端；当前实现没有自动创建
EC2、ECR、ALB、Batch 或 IAM 资源，也不会调整原有演示服务。

## 4. 验证

```bash
uv sync --locked --extra api --extra dev
uv run --extra api --extra dev pytest -q
uv run --extra api --extra dev ruff check backend finetuning scripts tests
uv run --extra api python -m scripts.verify_http_service
```

HTTP 测试覆盖认证、配置与路径限制、幂等并发提交、队列容量、真实 CLI 数据审计、
重启读取、质量失败分类、取消子进程、单 worker 锁、异常恢复及产物下载。
生命周期测试中的模拟子进程只验证任务状态和清理行为，不提供模型效果证据。
`scripts.verify_http_service` 在真实本机端口启动 API 和 worker，验证带认证的提交、
幂等、API 重启后队列保留、实际 CLI 审计、日志和产物下载；结束时清理所启动的进程。
它只验证 `validate` 模式，不运行 GPU 训练。
目前已验证 ARM API 镜像构建及两个容器中的真实数据审计；开发机为 ARM，
GPU 镜像的 x86_64 构建与新 HTTP 入口的完整 GPU 训练仍需在目标机器验收。
实际部署验证范围见 [验证记录](VERIFICATION.md)。
