# 插件契约总览

新增模块前先选边界。HarnessVLN 不要求继承抽象基类；只要对象满足 Protocol，并通过 YAML factory
构造，就能接入。结构类型降低依赖，但也意味着 contract test 必须承担运行时保证。

## 五类扩展点

| 扩展点 | 输入 | 输出/能力 | 生命周期 |
|---|---|---|---|
| `NavigationAgent` | `NavContext` | 主动函数调用并最终 stop | Task |
| `VLNNavigator` | Task + 受限 ToolClient | `vln.navigate.*` Job 工具 | Task / Run |
| `Environment` | Task；factory 可拿私有 Case | `nav.*` 工具、terminal、result | Task |
| `SpatialMemory` | Task + 受限 ToolClient | `spatial.*` 工具 | Task 实例，可持久化 |
| `Benchmark` | dataset | Case stream 与 score | Run |

## 稳定面与扩展面

**稳定面**：Protocol 方法、`NavTask/NavGoal/Observation/Pose`、工具名、现有 schema 语义、terminal
与清理顺序。

**扩展面**：`public`、`channels`、`extras`、requirements 映射、Environment result、provenance、
插件 params。扩展字段能快速实验，但一旦两个以上插件依赖同一语义，应提升为有文档和测试的协议。

## 接入前的四个问题

1. **谁决策？** 组织步骤、选择工具和恢复策略属于 Agent；动作预测内部循环属于 VLN。
2. **谁拥有原生对象？** simulator/service/robot 连接属于 Environment。
3. **谁拥有真值？** dataset loader 与 evaluator 属于 Bench。
4. **状态活多久？** Task 内状态放实例；重 checkpoint 可 run-scope；跨 Task 知识放 Memory 存储。

## 通用实现要求

### 构造

- factory 使用 `module:object`，参数可 YAML 化；
- Environment factory 接受 `case`，不要从全局读取“当前 episode”；
- 构造函数不应启动昂贵异步资源，启动放入 `start`；
- 参数错误尽早抛出具体异常。

### 能力声明

- Agent/VLN/Memory 的 `required_tools` 只列实际调用项；
- Environment 的 profile 精确写 channel、动作、frame、units 和 camera；
- VLN requirements 只写模型硬要求，不写偏好；
- 新 Tool 必须 `additionalProperties: false`，给写操作标记 `writes=True`。

### 生命周期

- 实例默认 single-use；
- `stop` 幂等，处理未完全 start、异常、超时和取消；
- 后台 task 保留引用并在 stop 中 await；
- 外部进程按进程组回收；
- stop 返回后不能再调用原生写操作。

### 可观测性

- 业务状态通过 Job status、terminal、result 或 tool result 返回；
- 版本身份写入 `provenance`；
- 错误包含 case/job/tool 上下文，但不吞原异常类别；
- 大型逐步数据留在 runs，Git 中提交紧凑、可校验 trace。

## 最小测试金字塔

| 层级 | 必测内容 |
|---|---|
| unit | 成功路径、非法参数、重复 stop、部分 start 失败、取消竞态 |
| contract | factory 可加载、profile/requirements、dataset 公私边界、固定资源版本 |
| integration | YAML -> stack -> 完整 Task -> score -> Manifest |
| real smoke | 真实 reset、观测、至少一个动作、goal finish、stop、无残留资源 |
| parity | 同版本官方 split/evaluator 指标对照 |

## 兼容性的判定

只有接口匹配不等于算法兼容。例如 R2R VLN 能调用 ObjectNav Environment 的 RGB-D 与离散动作，
但它没有类别目标训练和 prompt 语义，不能据此标记模型兼容。矩阵应分别记录：

- **mechanical compatibility**：工具/profile 可组合；
- **task compatibility**：输入目标范式与模型能力匹配；
- **validated compatibility**：真实 evidence 与 evaluator 对照。

## 何时修改核心

优先只新增插件。只有需求跨多个实现且无法用现有 tool/schema 表达时，才修改 ToolBus、runtime 或
公共 schema。核心改动必须补竞态测试与向后兼容说明，因为它的 blast radius 覆盖所有组合。
