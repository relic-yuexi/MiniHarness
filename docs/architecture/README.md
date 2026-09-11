# 架构题回答

以下每份文件回答一个模块的两道题，引用官方文档、源码或研究论文。调研日期：2026-09-12。回答中的长期记忆、分布式调度等是架构扩展，不代表最小实现已包含这些服务；当前实现以[项目 README](../../README.md)为准。

1. [Context / Performance](01-context-performance.md)：TTFT 与长对话压缩。
2. [Memory](02-memory.md)：召回策略、经典框架与评测。
3. [Task](03-task.md)：长任务目标保持、每天 9 点复盘调度。
4. [Tool / Session Runtime](04-tool-session-runtime.md)：异步工具、busy 队列与冲突。
5. [Runtime 架构对比](05-runtime-comparison.md)：三类工具协议、OpenHands 新旧架构。

原始设计草稿位于本地 `.local-docs/` 且不纳入 Git；这里是面试交付回答，不依赖本地草稿才能阅读。
