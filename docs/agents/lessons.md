# 经验与注意事项

## 文档和代码必须双向一致

文档里写“不支持”“仅支持”“必传”“预留参数不支持非空”等约束时，代码应有对应拦截；代码新增或收紧拦截时，文档应有同等语义说明。

检查时要双向走一遍：从文档逐条找代码拦截，从代码新增拦截逐条找文档说明。

## ABI 变更要显式说明

修改 `*_def.cpp`、`aclnn_*.h/.cpp`、PyTorch schema 或 opapi 适配代码时，可能影响 ABI 或调用兼容性。PR 中应说明影响范围、兼容策略和回归范围，并按 `.github/CODEOWNERS` 请求对应 owner 检视。

## 生成代码不要手改后遗忘来源

`torch_custom/fla_npu` 下部分文件由 YAML 和生成脚本产出。修改生成结果前，先确认是否应该改生成输入、生成脚本或模板。否则下一次生成可能覆盖手工修改。

## 旧调用路径只做兼容

先根据算子实现类型选择稳定 Python 入口：Ascend C 算子使用 `fla_npu.ops.ascendc`，Triton 算子使用 `fla_npu.ops.triton`，两者不是同一算子的必选双通路。`torch.ops.npu.*` 是可选 legacy 入口，只在实际实现时显式加载并做通路验证，不作为新增算子的主文档或主测试入口。

## varlen、layout 和可选状态最容易漏

线性注意力类算子经常同时支持 dense/varlen、多种 layout、不同 head 关系、可选 initial/final state 和特殊 gate 语义。新增 case 时优先覆盖这些分支的组合，而不是只测默认 shape。

## head ratio 不要隐式假设为 1

反向和局部梯度算子常见输出 head 数与 Q/K head 数不相等的场景。具体符号名使用所属模型根 README 的权威符号表；实现中应显式推导并校验 head ratio，用输出 head 映射回 Q/K head，测试中也要覆盖 ratio 大于 1 的情况。否则小 shape 或默认 head 数能过，但 GVA/grouped 场景会读错 head、写错 workspace slot 或复用错误中间结果。

## workspace slot 是协议，不只是地址

多阶段 AIC/AIV 算子里，workspace slot 同时承载数据和同步协议。slot 的数量、复用顺序、ready/free flag、tail chunk 空任务都必须匹配。不要让 AIV 跳过空任务而 AIC 仍等待，也不要让 AIC 覆盖 AIV 尚未消费的 tile。

还要区分 flag 计数器反压和 slot 所有权。`CrossCoreFlagWithReverse<Depth>` 的 reverse ACK 默认只说明 consumer 已执行一批 forward `Wait`；即使 `Depth` 恰好等于 slot 数，也不能证明异步读取已经结束。只有 FREE 发生在对应 slot 的最后一次消费 pipe 完成之后，才能允许 producer 复用。完整模型见 [`cross-core-pipeline.md`](cross-core-pipeline.md)。

## `<<<>>>` 通路的临时 GM buffer 要由框架托管

Fast Kernel Launch 里的 `<<<>>>` 调用是异步进入 NPU stream 的，`RunOpApi` 也只是把 launch 放入 `torch_npu` 的异步任务队列。workspace、tiling data 这类临时 GM buffer 不能按 `aclrtMalloc -> <<<>>> -> aclrtFree` 写法在 host 函数尾部释放；这会让 kernel 仍可能访问生命周期已经结束的地址，表现可能是 MTE 越界、静默错误，或在 `ASCEND_LAUNCH_BLOCKING=1` 下看起来正常。

框架层应统一使用 NPU Tensor 托管这类临时 buffer，并把 Tensor 本体捕获进 `RunOpApi`/`RunOpApiV2` 的 lambda。只保存 `GM_ADDR` 不够，因为 lambda 可能尚未执行，临时 Tensor 就已经析构。需要清零的系统 workspace 仍按当前 stream 做 `aclrtMemsetAsync`；kernel 从 GM 读取的 tiling struct 也应拷入 Tensor buffer，而不是单独申请 raw device 内存。

`ASCEND_LAUNCH_BLOCKING=1` 只能作为定位手段，不应作为正确性证明。修复应让 `ASCEND_LAUNCH_BLOCKING=0` 的异步路径本身拥有正确的 buffer 生命周期。

## Fast wrapper 和模板 kernel 的构建闭环

Fast wrapper 中如果直接使用 `aclrtMalloc` 分配 workspace、tiling data 或清零缓冲区，并在 `RunOpApi` 返回后立即 `aclrtFree`，异步 stream 仍可能在 host 函数返回后访问这些地址。应使用 NPU Tensor 托管临时 device buffer，并让执行 lambda 持有 Tensor 生命周期；修复后要在 `ASCEND_LAUNCH_BLOCKING=0` 下验证，`=1` 只能用于定位。

Fast example 的 CMake 还必须检查 `FAST_KERNEL_OP_NAME` 在当前 `NPU_ARCH` 下确实生成了源对象。当前 fast 源按平台目录组织，平台目录缺失时不能生成只有 `extension.cpp` 的空扩展并误报构建成功。CANN CCE linker 不接受部分新版本 CMake 注入的 `--dependency-file` 参数时，应关闭 linker 依赖文件选项，并保留 CMake 自己的依赖跟踪。

Ascend C 开启 `ASCENDC_TPL` 后，自动生成的 kernel metadata 会按模板参数实例化 kernel 入口。入口必须同时提供生成器要求的模板参数、默认 tiling 注册和默认 kernel type；不能保留一个非模板入口再让 metadata 以模板形式调用。dtype 模板参数应通过明确的 traits 映射到实际类型，且 A2/A5 都要至少完成算子编译验证。

Ascend C 算子构建应控制并行度。A5 主机核数可能远大于编译所需资源，直接使用 `os.cpu_count()` 或高并发构建可能被系统 OOM 杀死；fast wrapper 和算子构建都应提供可控的 job 数，并用串行或低并发构建复核真实编译错误。

## 提交前清理产物

构建目录、wheel、run 包、测试输出、profiling 输出、临时 patch、缓存和本地调测脚本不应进入提交。提交前查看 `git status --short`，确认未跟踪文件中没有生成物或敏感信息。
