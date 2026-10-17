# HTTP 服务验证

本次验证 HTTP 封装、持久队列、独立 worker 和容器部署链路。
内部继续使用 `finetune.py` 对应的 pipeline，未执行 GPU 训练或更新模型权重。

| 验证 | 结果 |
| --- | --- |
| 全部自动化测试 | 114 项通过；包含 21 项 HTTP/worker 测试、2 项进程组回归测试 |
| Ruff / diff 格式检查 | 通过 |
| 原生 HTTP | 真实 TCP 请求、认证、幂等、API 重启保留队列、实际 CLI 审计、日志与产物下载通过 |
| 数据 | `examples/choices.jsonl`，20 条，train/validation/calibration/test = 14/2/2/2 |
| Docker | ARM API 镜像构建成功；两个独立容器以 UID 10001 运行 API 和 worker，数据审计及重启查询通过 |
| Compose | v2.39.4 配置校验通过，校验工具下载 SHA-256 已核对 |
| GPU 镜像 | 目标为 Linux x86_64；当前 ARM 主机未完成该构建验收 |
| HTTP 完整 GPU 训练 | 未执行；已有 CLI GPU 实验不能替代此项验收 |

原始结果：[原生 HTTP](native-verification.json)、[容器 HTTP](container-verification.json)。
下载的数据清单与磁盘文件一致，SHA-256：
`3fa6be5de2463e83f7d7581534e885a2cafcd35da230b6e03ee7346171ded9c4`。

两个检查的最终状态均为 `succeeded`、`pipeline_status=data_validated`；
这仅代表数据检查完成，不代表微调效果或模型质量达标。
生命周期测试中的模拟进程用于验证退出码映射和取消，不用于产生训练指标。
测试过程中生成的 API token 未写入报告或仓库；验证进程和容器已清理。

复现命令及 AWS 部署方法见 [HTTP 服务文档](../../docs/HTTP_API.zh.md)。
