# 验证与边界

最后本地验证：2026-09-12，Windows，uv 0.8.9，Python 3.12.9。

- 离线测试：166 项通过；真实 API 测试 3 项明确跳过（无凭据）。
- Ruff 检查与格式检查通过。
- `uv build` 生成 wheel/sdist；隔离环境导入 wheel 并读取完整 compact prompt 成功。
- CLI `--version` / `--help` 正常。

远端 [Linux / Windows CI](https://github.com/relic-yuexi/MiniHarness/actions/runs/34631027807) 已全部通过（提交 `0b09fe3`）。后续最新状态可在仓库 Actions 查看。

## 覆盖范围

| 功能 | 可执行测试 |
| --- | --- |
| 配置、预算前置校验、Python入口 | test_config.py / test_cli.py |
| hash JSONL、断尾、损坏、归属、文件锁、未知写入 | test_storage.py |
| Schema、7工具、UTF-8/快照、文件并发与原子替换 | test_tools.py |
| 取消线程、管道关闭、已发布文件清理异常 | test_tool_edges.py |
| 三协议、流/非流、工具碎片、opaque、cache与retry用量 | test_providers.py |
| 主循环、追问、独立session、恢复、限额、steer/followup、compact | test_runtime.py |
| Hook持久化时机、启动/关闭竞态、compact期间输入、后台任务接受 | test_runtime_edges.py |
| Hook顺序与拒绝、完整工具配对、token估算 | test_hooks_context.py |
| 命名队列、覆盖/冻结/回滚、有效前缀恢复、Schema执行校验 | test_middleware_pipeline.py |
| 真HTTP、工具追问、todo重启、真实摘要 | test_live.py（显式opt-in） |

离线测试验证机制，真实模型是否按要求选择工具和压缩保真仍需 live 验收。测试不能证明面对任意输入都不会失败，也不能代替操作系统提供的持久性与进程边界。

## 明确限制

- 当前执行环境未提供 LLM API key，因此没有真实 API 成功记录；3 项 live 测试默认明确跳过。默认运行路径只有真实 HTTPProvider，没有关键词路由或伪造模型。
- 本地执行与测试为 Windows；远端 Linux/Windows 离线 CI 已通过。
- search 明确 mock；没有内置跨session向量RAG、定时复盘调度器、plan或/goal命令。
- 原生 `async: true` 工具调用属于扩展；当前跨Provider方案使用Job handle和独立完成事件。
- 同一session单进程owner；多session可并发。普通工具批次串行执行，后台Job并发。没有分布式worker或跨机器一致性协议。
- 文件工具防合作进程并发冲突，不承诺抵御恶意外部进程的TOCTOU攻击。Bash不是沙箱。
- 文件内容修改和日志没有共同事务；未知操作不自动重跑。Job只在本进程生存，崩溃恢复标unknown。
- `UTF-8 bytes/4`是用户指定的粗估；不是精确tokenizer。用量缺失时总计为已知下界，token停止阈值不是严格金额保证。
- JSONL当前启动时完整校验并加载事件；active context可压缩，但审计日志的内存/磁盘占用仍随历史增长。大型生产系统可再加索引/快照与流式回放。
- artifact二次读取以artifact自身为坐标；原始range来源信息应保留首次ref。没有额外的跨artifact provenance数据库。

## 重现

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv build
```

测试使用临时workspace，Bash测试只执行受控本地命令；live测试另建隔离session，不访问用户原有记录。参见README的真实API opt-in命令。
