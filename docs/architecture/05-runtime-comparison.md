# 模块五：Agent Runtime 架构对比

调研日期：2026-09-12。公开协议、产品行为和内部实现要分开回答；不把观察到的 Claude Code 界面当作其全部私有架构。

## 1. Claude Code 的工具输出，与 GLM/豆包 OpenAI-compatible function calling 有什么不同？

面试回答：先分三层：Claude Code 是完整 Agent 产品；Anthropic Messages、OpenAI Chat Completions/Responses 是模型调用协议；read/bash 的输出预览、引用、大结果截断则由应用 Runtime 决定。不能简单说“Claude Code 输出文本，国内模型输出 JSON”，两者都可使用结构化调用。

### 协议层比较

| 维度 | Anthropic Messages | OpenAI-compatible Chat Completions | OpenAI Responses |
| --- | --- | --- | --- |
| 工具定义 | name/description/input_schema | tools 中 function 的 name/description/parameters | function tool 定义，具体字段按 API 校验 |
| 调用 | assistant content 中 tool_use，input 为对象 | assistant.tool_calls，function.arguments 通常是 JSON 字符串 | output item 中 function_call，含 call_id/arguments |
| 结果 | user content 中 tool_result，以 tool_use_id 关联 | role=tool，以 tool_call_id 关联 | function_call_output，以 call_id 关联 |
| 文本与调用 | 有序 content blocks | content 与 tool_calls 字段分离 | 有序 typed items |
| 工具错误 | tool_result 有 is_error | 应用在结果内容中明确错误 | 应用在 output 中明确错误，按具体类型适配 |

Anthropic 的块模型天然表达文本、工具和多模态交错，但对结果位置严格；Chat Completions 的角色/调用分离容易接入已有 OpenAI SDK，但 arguments 仍需 JSON 解析、Schema 验证，兼容不代表所有参数完全一致；Responses 的 typed items 与 response 状态适合更丰富的交互，适配器必须保存需要回传的 item，而非只拼输出文本。这些是基于公开结构的工程权衡，不是服务质量排名。[Anthropic tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)、[OpenAI Function calling](https://developers.openai.com/api/docs/guides/function-calling)

GLM 官方 Function Calling 指南提供工具定义与调用结果回传流程，可归入 Chat Completions 风格；thinking 字段、Schema 子集、并行能力及模型约束仍须独立检查。[GLM Function Calling](https://docs.bigmodel.cn/cn/guide/capabilities/function-calling)

豆包方面，火山引擎官方 SDK 示例直接展示 `client.beta.chat.completions.parse/stream`、`message.tool_calls` 和 arguments delta，可确认其 SDK 的这一接入路径，不据此保证所有模型都支持相同能力。[火山引擎官方 parsing_tools.py](https://github.com/volcengine/volcengine-python-sdk/blob/master/volcenginesdkexamples/volcenginesdkarkruntime/parsing_tools.py) 本次 [Chat Completions 文档](https://www.volcengine.com/docs/82379/1494384) 与 Function Calling 文档页面抓取失败；因此具体模型的结果字段限制、原生 async 支持和完整参数矩阵保持未验证，不使用社区文章补成官方结论。

### 本项目如何吸收差异

内部 Message/Block/ToolCall/ToolResult 不直接等于某家 HTTP JSON；Provider adapter 负责投影与解析，并保留必要的 provider-specific opaque 字段。换 Provider 不是盲目重放旧签名 thinking 或专属 item；不兼容时明确限制或在完整边界重新摘要建立上下文。

流式时按 attempt 和 block/call index 累积文本与 arguments；块结束且整次输出完整验证前不执行工具。截断、拒绝和最终答案是不同结束原因。只记录 Provider 真正提供的 reasoning 摘要或必须回传的不透明块，不要求模型暴露完整内部思维链，也不靠 `<think>` 正则猜工具调用。

read 的 `{preview_content, ref_contents: []}` 是 MiniHarness 的结果契约，不是 Anthropic 或 GLM 的原生工具要求。preview 的 UTF-8 总字节数最多 32768，包含路径、前后片段和省略说明；ref 指向不可变完整快照并带范围/内容 hash。Runtime 首次生成结果即确定预览，后续只复用，不随源文件变化重写旧 message，兼顾缓存和可复现性。

测试用同一规范消息分别生成三种 payload，再将完整/分片响应恢复为同一语义；覆盖 Unicode 参数碎片、并行多个 call、空 content、重复 ID、错误 JSON、未知工具、thinking metadata、截断和缺失 usage。兼容端点必须有真实 API smoke test，不能只凭 SDK import 成功宣称兼容。

## 2. OpenHands 状态机的优缺点，更优雅的方式是什么？

先限定版本。旧版 OpenHands 0.54.0 的 AgentController 处理状态转换、事件、pending action、错误和预算，公开源码可见 `_pending_action` 以及异常情况下补结果的处理。显式状态和动作/观察关联有助于暂停、恢复、排查死循环；控制器同时承担多项职责时，增加状态和并发事件容易扩大条件组合与回归范围。这是针对该版本代码组织的评价，不是说 OpenHands 没有事件流。[OpenHands 0.54.0 AgentController](https://github.com/OpenHands/OpenHands/blob/0.54.0/openhands/controller/agent_controller.py)

新 Software Agent SDK 文档已经描述不可变 append-only EventLog、ConversationState、FIFO lock 和提交后回调。因此不能把“事件日志 + 单写者”包装成 OpenHands 从未有过的全新替代，也不能把旧版本局限无条件套给现在。[OpenHands SDK Conversation](https://docs.openhands.dev/sdk/arch/conversation)

对最小项目，更合适的不是彻底抛弃状态机，而是将状态机缩小为可测试的核心：

```text
command / worker event
        ↓
Session mailbox → single owner
        ↓
validate → append durable event → reducer(state, event)
        ↓
effect runner → Provider / Tool → result back to mailbox
```

reducer 只计算状态，不调用网络和工具；effect runner 执行 I/O，把结果转为事件；Context builder 从日志投影模型能接受的完整消息。Session、Turn、Step、Action、Job 各有小生命周期，不将 UI busy、工具执行、通知投递塞进一个巨大 enum。

优点是顺序确定、重放可测、失败有证据、Provider 与存储解耦；代价是事件版本迁移、日志增长、幂等、恢复和副作用不确定性必须认真实现。日志不自动解决事务：外部副作用不能与本地 append 原子提交；hash 链只能检测意外损坏，不能阻止有权修改全日志的人重算哈希。第一版明确单机单 session owner 比宣称支持分布式 exactly-once 更可靠。

验证重点是状态不变量：每个 committed tool call 最终有对应 result；只有 owner 写日志；内存等于已提交日志的归约；恢复不重新执行 effect；compact 不切断配对；重复事件不重复终结。用故障注入和顺序组合测试这些不变量，比只验证 happy path 的“模型叫了工具然后回答”更能说明 Runtime 设计质量。
