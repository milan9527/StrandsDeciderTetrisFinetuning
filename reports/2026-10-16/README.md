# 首轮实验记录

这些文件来自已完成的 GPU 训练任务，没有重新运行训练或修改评估结果。

- `training.json`：训练配置、源模型散列、验证曲线、环境及最终参数散列。
- `baseline/`、`adapted/`：两种检查点的测试统计和各 10 局 gzip 压缩回执。
- `summary.json`：对比统计；`replay-verification.json`：下载后的独立回放验证。
- `data/`：数据分区、种子、审计和 token 长度记录，完整训练数据可从引擎重新生成。
- `FILES.sha256.json`：本目录随仓库提供的文件散列。
- `SHA256.json`：原 S3 完整产出清单，包含未放入 Git 的模型权重；因此不应直接
  用它要求本目录拥有全部远程文件。

报告中的绝对路径为训练容器当时的路径，用于溯源。仓库内回放校验使用相对路径：

```bash
uv run python -m scripts.verify_reports
```

检查点在私有 S3 中，需已有读取权限。可将校准权重下载到本项目的忽略目录：

```bash
aws s3 sync \
  s3://strands-decider-tetris-training-632930644527/tetris-sft-20261016-102527/results/eval-v1/adapted/calibrated/ \
  artifacts/sft/aws-existing/job-v1/results/eval-v1/adapted/calibrated/
```

下载后需按完整清单验证权重，并按运行说明配置本机基座路径。
