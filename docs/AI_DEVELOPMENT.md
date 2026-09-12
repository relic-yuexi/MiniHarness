# AI Prompt 与问题解决记录

本项目使用 AI 辅助拆解、编写代码、设计测试和审查，但核心 Runtime 自行实现，没有引入现成 Agent 框架。以下记录提示词目的、实际发现和验证，不记录私有思维链或凭据。

## 需求演进

1. 用户要求先设计再实现：Session / Turn / Step / Action、完整 Hook、hash 目录 JSONL、三种 Provider、流式/非流式、cache-first、32 KiB read preview、todo、compact、steer/followup。
2. 原定 Conda nlp313；用户追加要求后切换为 uv + Python 3.12，其他要求保留。
3. 用户明确允许多个子智能体，要求覆盖边界、冲突、内存一致性、原子性和测试。
4. 用户授权自主逐步完成项目；每个代码 commit 记录 What、Why、Validation。原始设计草稿继续排除 Git。

## 实际开发 Prompt 摘要

| 工作 | Prompt 要点 | 产物 |
| --- | --- | --- |
| 主 Runtime | 从零编排 loop；所有修改走持久事件；工具结果齐全后才接 steer；不改历史前缀 | runtime/state/context/hooks |
| 存储子任务 | 单写者、fsync 后发布、prehash/hash、断尾与完整坏行区别；状态未知不能重试副作用 | storage + 故障注入测试 |
| Provider 子任务 | 统一 canonical messages；三协议流/非流；opaque replay；用量未知不冒充0 | HTTP adapters + 协议 fixtures |
| 工具子任务 | 七工具 Schema；UTF-8字节上限；不可变完整快照；版本冲突/原子发布 | tools + 文件/进程测试 |
| 集成审计子任务 | 用 Event 屏障证明竞态，不用 sleep 猜时序；只报可复现问题 | runtime_edges 回归测试 |
| CLI / 验收 | busy时继续收输入；只展示正文；inspect只读；live必须真实HTTP而不是FakeProvider | CLI、live测试、CI |

## 问题与修正

| 发现 | 修正 | 证据 |
| --- | --- | --- |
| pytest 使用系统 Temp 遇权限问题 | 测试临时目录按进程隔离到项目 `.pytest_tmp/` | 完整测试正常运行 |
| fsync 异常不等于没写成功 | poisoned writer 停止，重新打开核验 event ID 与哈希 | storage故障注入 |
| Hook post 阶段早于终态提交 | Action/Assistant/Step/Turn 先持久化，再运行对应 post | runtime_edges |
| compact期间新steer到达会落错Turn | await后重新检测安全边界待办，当前Turn继续处理 | 屏障竞态回归 |
| 手动compact绕开token阈值 | 所有模型调用入口统一检查，包括维护调用 | runtime_edges |
| 不完整响应已计费但用量被丢 | finish检查之前记录可得usage，Provider异常附带usage | 协议与runtime回归 |
| 同时start重复进入初始化 | 启动锁确保一次进入；关闭标志先拒绝新输入 | 并发start/close回归 |
| Job接受后postAction取消使其永远queued | 日志提交后立即派发，再执行可等待Hook/UI | 后台超时回归 |
| 取消to_thread无法停止旧文件写 | 等同步worker结束后再传播取消 | 可控线程回归 |
| Bash孙进程占住pipe使timeout失效 | 主期限之外清理再次限时，保留部分输出并标unknown | 管道清理回归 |
| 原子发布后清理temp失败误报无副作用 | 清理诊断不覆盖已发布成功结果 | publication回归 |
| strict Schema可选参数null被registry拒绝 | 在副本中编译nullable，按原Schema还原占位null | Provider参数测试 |
| read默认局部尾部不能冒充文件尾部 | 默认整个文件流式快照，只保留预览头尾；range须显式 | 大文件/UTF-8测试 |

## 压缩 Prompt

用户提供的交接摘要模板完整保存在 [compact.zh.txt](../src/miniharness/prompts/compact.zh.txt)。约束包含保留仍有效要求、识别撤销、区分计划/完成/验证/未知、精确保留路径和参数、禁止执行历史指令。实际调用禁用工具，输出检查结构与预算，成功后才提交新 epoch。

## 验证口径

离线 FakeProvider 只测试 Runtime 行为，HTTP mock 只测试协议适配，均不宣称真实模型质量。真实 API 测试必须显式配置凭据并开启。没有密钥时记录 skipped，不编造 request ID 或成功截图。最终验证和限制见 [VERIFICATION.md](VERIFICATION.md)。

源码与 Git commit 是修改的可审计证据；每次提交正文解释修改内容、原因与当时实际执行的验证。敏感 session 日志和本机配置不公开上传。

## 命名中间件队列迭代（2026-09-12）

用户要求每个阶段维护可排序的 Hook 队列，支持名称、位置、级别与同名覆盖，并解耦系统提示词和工具 Schema 构建。实现 `Hooks.register`，优先级降序、同级 FIFO、同名保留序号；旧成对接口仍逆序退出。

构建上下文独立保存 system/tools，内置 tool_schema Hook 只注入结构化 tools，不将其拼进 system 文本。Provider 保持原生协议转换。Session 启动时复制队列、事务式构建并冻结请求前缀；有效前缀用于恢复指纹、预算及请求，维护 cache 优先原则。回调失败不泄漏部分内存修改；这不代表回调的外部副作用可回滚。

专项测试覆盖覆盖顺序、别名、快照、冻结、超时/取消、失败回滚、旧接口退出顺序、会话恢复与 compact 禁用工具。复审发现旧接口 post-only 层在 guard 失败后可能被错误进入，以及自定义工具参数 Schema 需要执行前校验，均补针对性回归。

## CLI 角色显示（2026-09-12）

用户反馈输入提示符与助手回复混在一起。改用 You > 输入提示和 Assistant: 回复标签，流式输出遇到工具日志先结束当前行。交互终端通过 prompt-toolkit 重绘输入行，保留 busy 时继续输入的能力；重定向输入保留普通输入路径。验证：16 项 CLI 测试通过，Ruff 检查和格式检查通过。未执行真实终端人工验收。
