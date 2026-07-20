# 验证方法

算子测试资产、调用通路和交付矩阵以 [`../operator-development-standard.md`](../operator-development-standard.md) 为准。本文补充如何按风险组织验证，不另行定义测试归档位置。

## 验证分层

按改动风险选择验证范围，不要把“能编译”当作“功能正确”：

- 静态检查：`git diff --check`、schema/文档一致性检查、JSON 用例检查、生成物检查；Ascend C 算子同时检查 `--cce-auto-sync=off`。
- 环境检查：`python scripts/check_npu_env.py --build-only`。
- 构建验证：按 A2、A3、A5 目标 SOC 生成 wheel 或 OPP run 包。
- 单算子验证：从 `tests/op_cases/<op_name>.json` 读取用例，按 `tests/operators/<op_name>/README.md` 执行对应测试。
- 主精度验证：Ascend C 算子通过 `fla_npu.ops.ascendc`，Triton 算子通过 `fla_npu.ops.triton`，覆盖 JSON 中的精度、泛化、边界、功能分支和回归用例。
- 调用通路验证：Ascend C 算子覆盖 aclnn 与 `<<<>>>`；`torch.ops.npu` 仅在算子实现该可选入口时验证。
- 端到端验证：运行 `examples/flash_gated_delta_rule.py` 或 `ci/run_example_st_cases.py`。
- 精度验证：对比参考实现，覆盖关键 shape、dtype、layout、dense/varlen 和边界 case。
- 性能验证：使用合适 profiling 工具，不用 Python wall time 直接作为性能结论。
- 内存/同步验证：疑似越界、未初始化、流水 hazard 或同步问题时，使用对应 sanitizer/profiling 方法验证。

## 测试矩阵

测试矩阵应覆盖语义路径、调度路径和值域路径，而不是只堆几个默认 shape：

- 所有关键 shape、dtype、layout、属性、平台和用途标签先登记到 `tests/op_cases/<op_name>.json`；测试脚本不另建一套用例列表。
- 平台至少覆盖 A2、A3、A5，并分别包含基础功能和实现类型对应主入口的精度验证。

- fixed length 和 varlen 都要覆盖；varlen 场景要覆盖 `cu_seqlens` 与 `chunk_indices` 成对出现、尾 chunk、短序列和多 chunk。
- head 关系要覆盖一一对应和 GVA/grouped 场景，例如输出 head 数是 Q/K head 数的 1、2、4 倍，确认 head 映射和 workspace slot 没有串头；JSON 字段名使用所属模型根 README 规定的符号。
- 目标维度要覆盖关键模板组合，例如 `chunkSize=64/128`、`V=128/256`、主 dtype 为 `fp16/bf16`，以及 gate/scale 等辅助输入与主输入 dtype 不同的 mixed 场景。
- 可选但当前不支持的参数要有反向用例，确认代码会明确拦截，而不是静默忽略或在 kernel 内崩溃。
- 输出支持非连续视图时，要验证最终 `ViewCopy` 或等价路径；不要只测 contiguous 输出。
- 多阶段 AIC/AIV 协同算子要覆盖长序列、多 chunk、多 head ratio，让同一个 core 连续处理多个 task，分别触发 flag 深度反压与 workspace slot 复用协议；测试设计见 [`cross-core-pipeline.md`](cross-core-pipeline.md)。

## 构建矩阵

常用 SOC 映射：

- A2：`ascend910b`
- A3：`ascend910_93`
- A5：`ascend950`

修改公共接口、公共 kernel 组件或跨平台逻辑时，应考虑多 SOC 编译和必要运行验证。若当前环境无法覆盖某个 SOC，应在结果中明确说明未覆盖原因。

正式支持算子应覆盖 A2、A3、A5；无法覆盖某个平台时，还应按算子开发规范记录原因、用户影响、拦截行为和补齐计划。

## 精度问题处理

精度失败先分类，再决定处理方式：

- 如果误差呈结构性错位、整片符号/幅值异常、维度映射错误，优先回到索引、layout、任务分配、数据搬运和写回路径定位。
- 如果误差集中在无效区或 padding 区，先确认该区域的语义和测试后处理。
- 如果是随机数值误差，固定 shape/layout/属性后做多轮复检，再判断是否稳定劣于参考实现。

不要通过收窄输入 range、删除失败 case、降低覆盖强度或放宽阈值来制造通过结论。

## 结果记录

对外描述测试结果时，只写测试项和结果，不写本地机器、账号、绝对路径、临时目录或日志路径。结果至少能追溯到平台、JSON 用例总数、通过数和失败用例 ID。若没有执行某项验证，写清楚原因，例如缺少 NPU、缺少 CANN 环境或依赖版本不满足。
