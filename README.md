# MiniHarness

从零实现的 Python Agent：主循环、会话恢复、工具调度、流式解析、上下文压缩均由本项目实现，**不依赖 Agent 框架**。通过真实 HTTP API 对接 Anthropic Messages、OpenAI Chat Completions 和 OpenAI Responses，也可配置兼容端点。

Python **3.12**，使用 **uv** 管理解释器和锁定依赖。CLI 与 Python API 都可用。

## 快速运行

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后：

```bash
git clone https://github.com/relic-yuexi/MiniHarness.git
cd MiniHarness
uv sync --locked
cp config.example.toml config.toml
```

Windows PowerShell 使用 `Copy-Item config.example.toml config.toml`。编辑 `config.toml` 的 `provider.model`、`base_url`、`context_window` 和 `max_output_tokens`，按实际模型文档填写，不使用猜测的窗口大小。

设置配置中 `api_key_env` 指向的环境变量：

```bash
export OPENAI_API_KEY='你的 API key'
uv run miniharness doctor
uv run miniharness run '请用 calculator 计算 (17+25)*3，然后说明结果'
uv run miniharness chat
```

PowerShell：`$env:OPENAI_API_KEY = '你的 API key'`。密钥只从环境读取，不写入配置、日志或 Git；程序不自动读取 `.env`。`doctor` 只报告密钥是否存在，不请求模型。

```bash
uv run miniharness --config config.toml chat --user alice
uv run miniharness --config config.toml chat --user alice --session <64位session_id>
uv run miniharness run '你好' --no-stream
uv run miniharness sessions
uv run miniharness inspect <session_id>
```

每次未指定 `--session` 都创建新 session；同一个用户两个窗口也独立。启动时输出 session ID，可随时恢复。`inspect` 只读校验哈希链并展示事件、todo、用量和 Job 状态，不修复日志或执行工具。文件工具的 `workspace` 来自配置：显式使用同一路径的会话会共享文件，但不共享 todo、历史与队列。

交互命令：`/session`、`/todo`、`/compact`、`/cancel`、`/steer 新要求`、`/followup 下一轮问题`、`/help`、`/quit`。busy 时仍可输入，普通输入默认排到下一 Turn；`/quit` 等待已接收工作完成再关闭。`/new` 提示在新窗口不带 `--session` 创建独立会话。

## Provider 配置

| protocol | base_url 示例 | 凭据变量示例 |
| --- | --- | --- |
| `openai_chat` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `openai_responses` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic_messages` | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |

GLM、豆包等兼容端点按其文档填写 base URL、model 和参数。Chat 默认发送 `max_completion_tokens`；只接受旧字段的端点可在 `[provider.extra]` 设置 `max_tokens`。`strict_schema=true` 会编译 nullable 可选参数，不改变内部工具 Schema。协议兼容不等于所有模型参数完全兼容，需要针对实际端点执行 live 测试。

不支持当前会话中无损切换 Provider/模型/工具定义；恢复时配置指纹不一致会明确拒绝，另开会话即可。`--no-stream` 仅改变传输和展示，可用于恢复已有会话。

## 已实现的工具

| 工具 | 行为 |
| --- | --- |
| `calculator` | AST 白名单算术，有表达式/指数/结果大小限制，不执行任意代码 |
| `search` | 确定性示例资料检索，明确标记 **mock**，不冒充实时搜索 |
| `todo` | 当前 session 的 `list` / `replace`，用 `expected_revision` 防止旧状态覆盖 |
| `read` | UTF-8 文件、完整不可变快照或显式字节范围读取 |
| `write` | 默认仅创建；覆盖要求文件当前 hash；同目录临时文件原子发布 |
| `edit` | 当前 hash + 精确匹配次数校验，拒绝零/歧义匹配 |
| `bash` | 前台命令或 `background=true` 后台 Job，超时/输出限额与结果通知 |

每个工具注册名称、描述、JSON Schema；LLM 自主选择调用。参数严格解析，拒绝重复 JSON key、非有限数字和 Schema 不符输入。工具失败成为对应 tool result，让模型决定修正或结束。

`read` 的 `preview_content` **包含路径和省略提示在内不超过 32768 UTF-8 bytes**；默认展示整个文件的头尾。`ref_contents` 指向内容寻址的不可变 artifact。通过 `artifact_id`、`offset`、`limit` 可读取省略部分，偏移相对于当前 artifact；显式范围读取的首次结果记录源范围，二次 artifact 读取以该快照自身为坐标。默认单文件最大 16 MiB，完整快照流式生成；大文件不会因预览而全部驻留内存。

文件工具限制在 workspace 内，并拒绝链接/路径越界及覆盖运行时存储；共享文件由路径锁和版本 hash 防止合作进程间丢失更新。Bash 拥有当前账户的实际执行权限，**不是安全沙箱**。Windows 需 Git Bash 等真实 Bash；可以配置 `bash_executable = "C:/Program Files/Git/bin/bash.exe"`，不会偷偷改用 PowerShell。输出每管道最多保留 8 MiB，预览有界；无法确认副作用时标记 `unknown`，不自动重试。

## 系统设计

```text
CLI / Python API → durable queue → Runtime owner
                                  ├─ Context → HTTP Provider → Completion
                                  ├─ ToolRegistry → tool / background job
                                  ├─ Hooks
                                  └─ SessionStore → session.jsonl → pure reducer
```

- **Session**：一个窗口的独立空间，目录为随机 nonce 与 user ID 的 SHA-256。
- **Turn**：一次正常输入到结束，期间可加入 steer；followup 开新 Turn。
- **Step**：一次模型决策与它发出的全部工具调用；默认每 Turn 最多 16 Step。
- **Action**：一次工具调用；内部 action ID 与 Provider tool call ID 分开。
- **Job**：后台工作，提交 Action 返回 handle 即闭合；完成事件不构成原调用第二份 tool result。

Runtime 自行判断继续/结束，不使用关键词规则替代模型选择。普通工具批次串行执行，保证文件和 todo 的确定性；await 期间入口仍接收消息，后台 Job 可并发。最多 16 个运行中 Job；用户队列默认 128 条，满时拒绝且不回成功回执。

`steer()` 等已发出的工具批次完整闭合后加入当前 Turn。`followup()` 在该 Turn 后处理。序号决定消息与通知顺序，不依赖时间戳。取消不等于撤销已发生的副作用；文件 worker 先收敛再允许后续任务，避免取消后仍有旧写入。

每个 Hook 位置维护命名中间件队列，通过 `hooks.register(name, callback, position=..., priority=0)` 注册异步回调。优先级数值越大越先执行，同级按注册顺序；名称在同一注册表内唯一，同名覆盖并保留原排序序号，也可以改变位置。自定义同名 Hook 覆盖内置实现，与注册先后无关。位置兼容 `preActionHook`、`pre_action_hook`、`pre_action` 写法。

`pre_system_prompt` / `post_system_prompt` 是可变的构建管线，接收 `{"system": ..., "tools": [...]}`；回调可修改副本或返回 dict 补丁。内置 `tool_schema` 在 `post_system_prompt` 注入工具 Schema，可同名替换。Schema 仍通过 Provider 原生 tools 字段发送，不重复拼进 system 文本；工具名称必须已有注册的执行器，替换 Schema 不会替换执行器及其参数校验。

```python
from miniharness.hooks import Hooks

hooks = Hooks()


async def add_policy(context):
    context["system"] += "\n修改文件后说明验证结果。"


hooks.register("project_policy", add_policy, position="pre_system_prompt", priority=10)
# 将 hooks 传给 Runtime(..., hooks=hooks)。
```

每个 Session 复制注册表，在启动时执行构建管线，再冻结这两个位置。有效 system/tools 共同参与恢复指纹、请求与 token 预算；中途不能修改当前 Session 的缓存前缀。compact 请求仍禁用工具。执行队列使用快照，运行中注册只影响下一次派发。构建回调失败时丢弃其修改，guard 失败终止构建；observer 错误记录后继续。生命周期位置（session/user/turn/step/assistant/action/compact）的历史数据只读，不能绕开日志修改状态。默认回调超时 5 秒。

兼容接口 `Hooks.add(scope, pre=..., post=..., guard=True)` 保留成对 pre 顺序进入、post 逆序退出的语义；新接口的 pre/post 均按优先级/FIFO。post 错误记录但不撤销已持久化操作；Hook 对外部资源的副作用不能自动回滚。

### 持久化与一致性

```text
.miniharness/sessions/<session_hash>/
  session.jsonl
  session.lock
  artifacts/<sha256>
```

每条事件包含 schema_version、seq、timestamp、event_id、session/turn/step/action ID、causation_id、prehash/hash。先 append + flush + fsync，再更新内存 reducer，最后确认接受。todo 的新版本与 Action 结果在同一事件中提交。

写盘状态不明或 reducer 失败时 owner 停止，重开校验并重放，不继续使用过期内存。无换行尾部先备份再修复；完整坏行停止，不能跳过。hash 链用于定位和检测损坏，不是防篡改数字签名。启动时重放不会再次执行工具；已开始无结果的操作标记 unknown。

文件系统副作用和 JSONL 不能组成同一事务，不承诺 exactly-once。后台 worker 是进程内任务；进程崩溃后状态 unknown，不假装继续运行，也不盲目重跑。UI 完成通知即时展示，持久化通知在下一安全边界进入模型上下文。

### 流式与用量

流式 delta 批量写 trace，未完成输出仅供展示；参数与完整响应通过终态检查后才提交和执行。中断、拒答、截断与正常答案区分。保存 Provider 公开 reasoning 字段/必要 opaque blocks，不要求或伪造私有思维链。

记录 input/output、cache read/write、reasoning 与 raw usage；累计快照不能重复相加。未知为 null；失败重试的已知用量计入，未知部分标记。`total_tokens` 是已知用量下界，`max_session_tokens` 是停止后续调用的阈值，不是严格金额上限；不能据此保证一次请求不越过预算。

### Context、compact 与 memory

模型看到固定 system/tools + 当前 epoch 消息；日志/usage/临时流片段不全塞进 prompt。非 compact 操作只追加，旧消息与 Schema 不变，缓存命中仍取决于服务端。

手动 `/compact` 与自动阈值触发。文本估算为 `ceil(UTF-8 bytes/4)`，包含工具和协议包装；同时预留普通输出、压缩 prompt（至少 2K）、摘要（默认 8K）和可配 reasoning 预算，按窗口的 80% 做保护。估算不等于 tokenizer，超大单条输入会有界失败，不无限压缩。

compact 在完整工具边界建立快照，调用同一真实 Provider、禁用工具；摘要必须完整且包含任务/状态结构。新 epoch 用摘要替换旧前缀，原 JSONL 不改。期间到达输入仍入队；失败保留旧 epoch。默认保留最近两个完整 Turn；单个长 Turn 可在已闭合 Step 边界整体压缩。

**Memory 召回时机与放置**：恢复时从日志还原 active messages、摘要、todo；普通追问使用已有 history，工具结果作为对应 tool message。todo 更新通过工具结果追加，compact 额外放入权威 todo 快照。旧摘要与新证据合并，保留仍有效需求、撤销关系及未知动作。首版没有跨 session 向量 RAG，不会自动把窗口一的记忆带入窗口二。长期 memory、定时复盘、plan、/goal 和原生 async function calling 的扩展设计见架构回答。

## Python API

```python
import asyncio
from miniharness.config import load_config
from miniharness.runtime import Runtime


async def main():
    async with Runtime(load_config(), user_id="alice") as agent:
        ticket = await agent.submit("使用工具计算 19*23")
        correction = await agent.steer("请简短回答")
        print(await ticket.wait())
        print(await correction.wait())
        print(agent.session_id)


asyncio.run(main())
```

一个 Runtime 仅由一个事件循环拥有，同 session 使用跨进程文件锁排他；不同 session 可分别运行。持久化 `request_id` 支持重试去重，相同 ID 不同内容明确冲突。

## 测试与真实 API 验收

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv build
```

离线测试使用可控 Provider/HTTP fixtures 检查编排和协议，不冒充真实模型。覆盖三协议、流式碎片、工具安全/原子修改、session 恢复、fsync 故障、队列竞态、Hook、compact 和后台取消。临时测试文件位于 `.pytest_tmp/`。GitHub Actions 在 Linux/Windows 跑离线测试。

真实 API 测试默认跳过；配置好凭据后显式执行（会消耗 API 额度）：

```bash
MINIHARNESS_LIVE=1 MINIHARNESS_CONFIG=config.toml uv run pytest -m live -v
```

PowerShell：先设置 `$env:MINIHARNESS_LIVE='1'` 和 `$env:MINIHARNESS_CONFIG='config.toml'`，再运行 `uv run pytest -m live -v`。覆盖真实流/非流、工具追问、todo 重启、真实摘要与纯对话记忆追问。缺凭据/配置会明确 skip；结果应按实际运行报告，不把 skip 当通过。

## 提交材料

- [五个架构模块回答与信源](docs/architecture/README.md)
- [AI Prompt 与问题解决记录](docs/AI_DEVELOPMENT.md)
- [测试与实现边界](docs/VERIFICATION.md)
- 原始本地设计文档保留在 `.local-docs/`，按要求不纳入 Git。仓库中的架构题回答是单独的交付版本。

每个实现 commit 都包含 **What / Why / Validation**。API key、运行 session、私有配置和测试临时数据不提交。
