# 模块四：Tool / Session Runtime

调研日期：2026-09-12。术语与 [Runtime 设计](../../README.md) 一致：Session 为独立窗口；Turn 是一次用户任务处理；Step 为一次模型决策及其工具批次；Action 是一次具体工具调用；Job 是可跨 Turn 存活的后台工作。

## 1. 异步工具不能一直等，怎样执行与通知？

先区分四件事，不能用同一个“异步”混称：

| 机制 | 实际解决的问题 | 本项目策略 |
| --- | --- | --- |
| Python asyncio / 并发工具 worker | 客户端不阻塞、独立调用并发 | 可以实现，与协议是否允许缺结果无关 |
| OpenAI background mode | 一个模型响应在服务端后台生成 | 不等于一个外部工具可以缺结果继续推理 |
| Provider 原生 async tool calling | 协议允许工具未返回时继续独立工作 | Capability 显式开启，首版不依赖 |
| 应用层 Job + handle | 先完成“提交任务”调用，后台继续做工作 | 跨 Provider 的默认方式 |

OpenAI 官方页面目前明确 GPT-6 Astra and later 支持 async tool calling；对应用运行的 function/custom tools 生效，不能据此泛化到所有模型、托管工具和兼容端点。兼容列表是一份动态能力约束，不硬编码“所有 OpenAI 模型都支持”或“OpenAI 从来不支持”。[Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling) Background 则是响应生命周期能力。[Background mode](https://developers.openai.com/api/docs/guides/background)

### 默认的 Job 模式

机制示例：LLM 提交后台任务，Runtime 持久化 job intent 和参数、生成 job_id；提交动作的工具结果立即返回 `{status: accepted, job_id: ...}`。这个 Action 到此已完成，不能向用户宣称后台工作也完成。worker 执行后将 `job.completed/failed` 事件发回 owner；owner 写盘、创建通知，下一安全边界将其追加为有来源的后台结果消息，或由查询操作读取。后来的结果不是给原 tool_call_id 填第二个 tool result。这里的提交/查询是语义示意，不要求新增 start_job/get_job 工具；可采用 bash 的 background 模式，具体接口留待 I7 确定。

Job 生命周期为 queued/running/succeeded/failed/cancelled/unknown；通知独立跟踪 created/delivered。UI 断开不取消工作。工具完成与通知发送分离，通知失败不重跑副作用。基础版进程内 worker 在进程终止后不能承诺继续运行，恢复记 unknown/lost 并如实告知；后续若增加独立 worker，心跳或租约失效也只表示失联，不自动证明任务失败。只对幂等工作自动重领，副作用结果不明则查询外部状态或转 unknown。通知使用稳定 notification_id 和持久化 outbox，重试可能重复投递，接收端去重；不承诺不支持幂等的外部渠道 exactly-once。用户取消只阻止尚未开始的动作或请求协作取消；操作已到外部系统，不能承诺撤回。

对支持原生 async 的适配器，可让 Action pending 并按该 Provider 的协议继续；它是不同的投影策略，不能绕过基础版“工具配对完整”的校验器。当前项目用 Job handle 保持三类 Provider 一致，避免为了高级特性污染通用 Runtime。

## 2. session busy 时，新消息和异步结果同时到达怎么办？

采用每 session 一个 owner 加 mailbox。入口持续接收请求、分配 `request_id/enqueue_seq` 并持久化；只有 owner 能修改 ContextView、分配日志 seq 或启动下一次 LLM。不同 session 可以并行；同一 session 不允许两个 owner，也不能只靠进程内 asyncio.Lock 处理多进程写盘。

`busy` 只是对外展示状态；内部还要有当前 Turn/Step、pending Action 集合、queue 和 budget，不能一个布尔值承载全部运行语义。

### steer 与 followup 的确定语义

steer 的“当前 Turn 最后一个 Action 后”定义为：当前已发出 Step 的整批 Action 全部形成终态结果后，在下一次模型请求前追加 user，并继续同一 Turn。一个 Step 有多个工具时不能在第一个工具完成后插入；必须闭合整个调用批次。

followup 在当前 Turn 结束后启动新 Turn，默认普通输入走 followup。纯文本响应生成期间到达 steer 时，先完整 commit assistant，再消费 steer 并继续 Step；若 Turn 已经原子关闭，则 steer 降级为下个 Turn，回执告知实际落点。steer 不修改已发送给 Provider 的请求，不保证能撤销已经启动的副作用，也不重置本 Turn 的 step 限制。

例如：

```text
Step 1 assistant: call A, call B
队列到达: steer U1，job J 完成，followup U2
A 完成；B 完成
追加 results A/B（按调用顺序投影）
追加 U1 与 J 通知（按入队顺序，仍在当前 Turn）
Step 2 assistant ... → Turn 结束
接受 U2 → 新 Turn
```

磁盘上的工具完成事件保留实际到达顺序；发给模型的结果按原调用顺序组织。Anthropic 文档要求 tool_result 紧随对应 tool_use，且同一 user 消息中的结果块排在普通文本前，因此适配器合并结果与 steer 文本时也要校验块顺序。[Anthropic Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)

### 冲突、原子性与恢复

1. 用户消息与 job 完成按 owner 分配的单调序号排序，不依赖不同机器的时间戳。两者都先落盘再确认接受；客户端用 request_id 重试不重复插入。
2. 每个调用都必须有一个结果，包括 Schema 错、拒绝、超时、取消和结果未知；不能因其中一个失败就漏掉同批其他结果。重复 tool_call_id 在解析阶段拒绝；内部 action_id 与 Provider call_id 分开。
3. 外部副作用和本地 JSONL 无法原子提交。先记 started 再执行，crash 后发现 started 无 completed 标为 unknown，不能自动重跑 write/bash。优先用外部幂等键、查询执行状态；无法确定则向用户报告。
4. 日志 append/flush/fsync 成功后 reducer 更新内存。写失败时提交可能未知，将 store 标记 poisoned 并停止推进，不凭旧 RAM 继续，也不立即重试同一副作用；重新打开、校验并重放日志后确定持久状态。不能回复“已保存”后只存在 RAM。恢复逐行验证 seq/prehash/hash；末尾半行可隔离修复，中间损坏停止。重放不触发工具或 Hook。
5. compact 只在安全边界提交新 epoch；期间 ingress 持续收件，但不并发修改被摘要的源。post Hook 失败记诊断，不回滚已经提交的工具，也不二次执行。
6. 用户队列初值上限 128，满时返回明确“未接受”的可重试背压响应，不能确认接收后丢弃。控制事件与 worker 结果保留容量或写入独立持久化入口，已执行动作的结果不能因普通消息塞满被丢弃。持续 steer 不能让 followup 永远饥饿。cancel 作为控制事件阻止新副作用，但已到外部的工作仍如实追踪。迟到结果带原 job/action ID，不冒充当前任务的成功。

### 关键测试

用受控 Future 而非 sleep 构造竞态：工具 B 先完成、两个 steer、一个 followup、同时 job 结果；断言发往 Provider 的每个请求都完整配对、顺序确定且 Turn ID 正确。覆盖 close 与 enqueue 竞态、取消与工具完成竞态、重复完成事件、重启恢复队列、磁盘失败、尾行损坏、Hook 异常、流断开及 unknown 副作用不重放。所有 Provider 都跑同一契约测试，再分别测消息投影。
