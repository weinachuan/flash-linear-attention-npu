// 仅伪代码。FwdH kernel 入口和 sequence -> round -> chunk 总调度。
// 本文件对应真实 op_kernel 中的 chunk_gated_delta_rule_fwd_h.cpp。
// 伪代码目录不加入任何 CMake 构建目标。

#include "chunk_gated_delta_rule_fwd_h_policy.h"
#include "chunk_gated_delta_rule_fwd_h_utils.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_gated_delta_rule_fwd_h_cube.h"
#include "arch35/chunk_gated_delta_rule_fwd_h_vec.h"
#else
#include "arch22/chunk_gated_delta_rule_fwd_h_cube.h"
#include "arch22/chunk_gated_delta_rule_fwd_h_vec.h"
#endif

#include "chunk_gated_delta_rule_fwd_h_tiling_key.h"
#include "lib/matmul_intf.h"

// 真实入口由 kernel_operator.h 提供 GM_ADDR、__global__ 和核函数调度宏。
#include "kernel_operator.h"

namespace fwd_h_pseudocode {

enum class KernelRole { Cube, Vector };

inline CubeStageResult RunStage0ByArch(const CubeStage0Args& args)
{
    // Stage0 分发公式：P_c,h = W_c,h @ H_c,h；按目标架构选择对应 Cube 实现。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return RunStage0Arch35(args);
#else
    return RunStage0Arch22(args);
#endif
}

inline VecStageResult RunStage1ByArch(const VecStageArgs& args)
{
    // Stage1 分发公式：V_new_c,h = cast_BF16(fp32(U_c,h)-fp32(P_c,h))（无 P 时取零）；按目标架构选择对应 Vec 实现。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return RunStage1Arch35(args);
#else
    return RunStage1Arch22(args);
#endif
}

inline CubeStageResult RunStage2ByArch(const CubeStage2Args& args)
{
    // Stage2 分发公式：g-only 为 D_c,h = k_raw_c,kh^T @ V_new_g,c,h，gk-only 为 D_c,h = kg_c,kh^T @ V_new_c,h。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return RunStage2Arch35(args);
#else
    return RunStage2Arch22(args);
#endif
}

inline VecStageResult RunStage3ByArch(const VecStage3Args& args)
{
    // Stage3 分发公式：R_{c+1,h} = gate(R_c,h) + D_c,h；按目标架构选择对应 Vec 实现。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return RunStage3Arch35(args);
#else
    return RunStage3Arch22(args);
#endif
}

inline void RunSMinusOneByArch(const SMinusOneArgs& args)
{
    // S-1 分发公式：H_0,h = cast_BF16(layout_decode(initial_state_h))；按目标架构选择对应 Vec 实现。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    RunSMinusOneArch35(args);
#else
    RunSMinusOneArch22(args);
#endif
}

inline void RunOneChunk(SchedulerContext& ctx, const RoundPlan& plan, KernelRole role)
{
    // 单 chunk 阶段组合公式：S0 为 P=W@H，S1 为 V_new=U-P，S2 按分支计算 D，S3 为 R_next=gate(R)+D。
    if (role == KernelRole::Cube) {
        if (FwdHStagePolicy::NeedStage0(plan)) {
            // S0 只完成 H/W MTE2、MTE1、MMAD 和架构对应的 P 写回，并发布 PReady；不搬运 kg。
            RunStage0ByArch({&ctx.inputs, &ctx.outputs, &ctx.tiling, &plan, &ctx.workspace,
                             &ctx.memory, &ctx.sync});
        }
        if (FwdHStagePolicy::NeedStage2(plan)) {
            // S2 先按 requiredKh[] 搬运 kg，再完成右操作数转 NZ、MMAD 和 D 写回。
            RunStage2ByArch({&ctx.inputs, &ctx.tiling, &plan, &ctx.workspace, &ctx.memory, &ctx.sync});
        }
        return;
    }

    const bool finalVNewOnly = plan.finalVNewOnly;
    const bool noInitialFirst = plan.chunk.first && !ctx.tiling.useInitialState;
    Stage1Variant variant = Stage1Variant::WithP;
    if (finalVNewOnly) {
        variant = Stage1Variant::v_new_only;
    } else if (noInitialFirst) {
        variant = Stage1Variant::NoP;
    }
    // S1 等待 PReady（如有），一次 VF 生成 V_new/右操作数；右操作数先经 MTE3 写 GM ND，
    // 由 S2 再搬成 L1 NZ 后才发布 RightL1Ready。
    RunStage1ByArch({&ctx.inputs, &ctx.outputs, &ctx.workspace, &ctx.tiling, &plan,
                     &ctx.memory, &ctx.sync, variant});

    if (FwdHStagePolicy::NeedStage2(plan)) {
        // S3 等待 DReady，更新 rolling state；末 chunk 按需写 final_state。
        RunStage3ByArch({&ctx.inputs, &ctx.outputs, &ctx.tiling, &plan, &ctx.memory, &ctx.sync});
    }
}

inline void RunFwdH(SchedulerContext& ctx, KernelRole role)
{
    // FwdH 总体公式：按 sequence -> head_round -> chunk 顺序递推 D_c,h（由 V_new_g/V_new 分支计算），R_{c+1,h}=gate(R_c,h)+D_c,h。
    // Host 校验已经建立全部 shape/dtype/layout 不变量。
    for (int n = 0; n < ctx.tiling.sequenceCount; ++n) {
        const auto& seq = ctx.tiling.sequences[n];
        // 循环顺序固定为：sequence -> head_round -> chunk。
        for (int round = 0; round * kMaxRoundHeads < ctx.tiling.hv; ++round) {
            // 先固定本 head_round 的 requiredKhCount 和每个 H_v -> kgSlot 映射。
            // 这些关系只依赖 HK/HV，与 chunk 无关；后续 chunk 只绑定 token payload。
            const RoundPlan headRoundPlan = BuildHeadRoundPlan(ctx.tiling, round);
            if (round > 0 && role == KernelRole::Vector) {
                // 上一 round 的 kg、H、W 及异步搬运全部排空后，才允许本 round 发起搬运。
                const RoundPlan previousHeadRound = BuildHeadRoundPlan(ctx.tiling, round - 1);
                const RoundPlan previousRound =
                    BuildChunkPlan(previousHeadRound, ctx.tiling, seq, seq.chunkCount - 1);
                ctx.sync.WaitBeforeNextRound(previousRound);
                ctx.memory.ReleaseStateAfterRoundBarrier(previousRound);
            }

            if (role == KernelRole::Vector) {
                // S-1 是当前 round 的生产者，必须在屏障后执行，并在首个 S0 前排空。
                RunSMinusOneByArch({&ctx.inputs, &ctx.outputs, &ctx.tiling, &seq,
                                    &ctx.memory, &ctx.sync, headRoundPlan.activeHvBegin,
                                    headRoundPlan.activeHvCount, round > 0});
            }

            for (int c = 0; c < seq.chunkCount; ++c) {
                // chunk 只绑定 token 范围和最终 chunk 分支，不重新计算 required_hk_round。
                const RoundPlan plan = BuildChunkPlan(headRoundPlan, ctx.tiling, seq, c);
                RunOneChunk(ctx, plan, role);
            }
            if (role == KernelRole::Vector) {
                ctx.sync.Set(EventKind::TerminalDrain, round, /*round 生产者*/ 1, /*调度器*/ -1);
            }
        }
    }
}

// 设备 kernel 只接收 GM 地址和 host 已生成的 tiling。参数校验、输出分配和 workspace 大小
// 均在 op_host 完成；这里仅把地址绑定到阶段调度上下文，不能在 kernel 内重新做 host 适配。
inline void BindTensor(TensorRef& tensor, GM_ADDR address)
{
    tensor.gm = address;
    tensor.present = address != nullptr;
}

inline TilingPlan DecodeKernelTiling(GM_ADDR tiling)
{
    // 实现时按 ChunkGatedDeltaRuleFwdHTilingData 的字段顺序从 GM tiling 解码；不重新推导 shape。
    (void)tiling;
    return {};
}

inline WorkspaceRefs BindKernelWorkspace(GM_ADDR user, const TilingPlan& plan)
{
    // user 指向 AscendC::GetUserWorkspace(workspace)；各 scratch 偏移由 op_host tiling 给出。
    (void)user;
    (void)plan;
    return {};
}

inline void BindKernelContext(SchedulerContext& ctx, GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                              GM_ADDR initialState, GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState,
                              GM_ADDR user, GM_ADDR tiling)
{
    ctx = SchedulerContext{};
    BindTensor(ctx.inputs.k, k);
    BindTensor(ctx.inputs.w, w);
    BindTensor(ctx.inputs.u, u);
    BindTensor(ctx.inputs.g, g);
    BindTensor(ctx.inputs.gk, gk);
    BindTensor(ctx.inputs.initialState, initialState);
    BindTensor(ctx.outputs.h, h);
    BindTensor(ctx.outputs.v_new, vNew);
    BindTensor(ctx.outputs.finalState, finalState);
    ctx.tiling = DecodeKernelTiling(tiling);
    ctx.workspace = BindKernelWorkspace(user, ctx.tiling);
}

struct KernelDispatchTiling {
    int64_t dataType = 1;      // 0: FP16，1: BF16
    int64_t gDataType = 1;     // 1: BF16，2: FP32
    int64_t stateDataType = 1; // 1: BF16，2: FP32
    bool useGk = false;
    bool useExp2 = false;
};

inline KernelDispatchTiling ReadKernelDispatchTiling(GM_ADDR tiling)
{
    // 这里按 op_host 的 ChunkGatedDeltaRuleFwdHTilingData 读取运行期 dispatch 字段；
    // 所有运行期分支均发生在 VF 之外，VF 本身只接收编译期模板参数。
    (void)tiling;
    return {};
}

} // namespace fwd_h_pseudocode

namespace GDN {

using namespace fwd_h_pseudocode;

constexpr uint32_t kTilingKeyV128 = 1;
constexpr uint32_t kTilingKeyV256 = 2;
constexpr uint32_t kGateG = 0;
constexpr uint32_t kGateGk = 1;
constexpr uint32_t kExpModeNatural = 0;

struct Fp16 {
};
struct Bf16 {
};
struct Fp32 {
};

struct TileShapes128 {
    static constexpr int valueDim = 128;
};
struct TileShapes256 {
    static constexpr int valueDim = 256;
};

template <uint32_t VTile>
struct FwdHTileSelector;

template <>
struct FwdHTileSelector<kTilingKeyV128> {
    using type = TileShapes128;
};

template <>
struct FwdHTileSelector<kTilingKeyV256> {
    using type = TileShapes256;
};

template <uint32_t GateMode, typename GateElement>
struct FwdHGateTypeSelector;

template <typename GateElement>
struct FwdHGateTypeSelector<kGateG, GateElement> {
    using type = GateElement;
};

template <typename GateElement>
struct FwdHGateTypeSelector<kGateGk, GateElement> {
    using type = GateElement;
};

template <typename InputT, typename WorkspaceT, typename TileShapes, uint32_t kGateMode>
class ChunkGatedDeltaRuleFwdHCube {
public:
    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR h, GM_ADDR cuSeqlens,
                                GM_ADDR chunkIndices, GM_ADDR user, GM_ADDR tiling)
    {
        BindKernelContext(ctx_, k, w, nullptr, nullptr, nullptr, nullptr, h, nullptr, nullptr, user, tiling);
        ctx_.tiling.valueDim = TileShapes::valueDim;
        ctx_.tiling.gateMode = kGateMode == kGateG ? fwd_h_pseudocode::GateMode::ScalarG
                                                   : fwd_h_pseudocode::GateMode::KeyWiseGk;
        (void)cuSeqlens;
        (void)chunkIndices;
    }

    __aicore__ inline void Process()
    {
        // AIC 只执行 Stage0/Stage2；Stage1/Stage3 的 ready/free 由 AIV 通过同步台账交接。
        RunFwdH(ctx_, KernelRole::Cube);
    }

private:
    SchedulerContext ctx_{};
};

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          typename TileShapes, uint32_t kGateMode, uint32_t ExpMode>
class ChunkGatedDeltaRuleFwdHVector {
public:
    __aicore__ inline void Init(GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR vNew,
                                GM_ADDR finalState, GM_ADDR user, GM_ADDR tiling)
    {
        BindKernelContext(ctx_, nullptr, nullptr, u, g, gk, initialState, h, vNew, finalState, user, tiling);
        ctx_.tiling.valueDim = TileShapes::valueDim;
        ctx_.tiling.gateMode = kGateMode == kGateG ? fwd_h_pseudocode::GateMode::ScalarG
                                                   : fwd_h_pseudocode::GateMode::KeyWiseGk;
        ctx_.tiling.useExp2 = ExpMode != kExpModeNatural;
        (void)cuSeqlens;
        (void)chunkIndices;
    }

    __aicore__ inline void Process()
    {
        // AIV 执行 S-1、Stage1 和 Stage3；所有 VF 入口在各自 arch 文件中完成。
        RunFwdH(ctx_, KernelRole::Vector);
    }

private:
    SchedulerContext ctx_{};
};

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          typename TileShapes, uint32_t kGateMode, uint32_t ExpMode>
class ChunkGatedDeltaRuleFwdHKernel {
public:
    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                                GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState,
                                GM_ADDR user, GM_ADDR tiling)
    {
        cube_.Init(k, w, h, cuSeqlens, chunkIndices, user, tiling);
        vector_.Init(u, g, gk, initialState, cuSeqlens, chunkIndices,
                     h, vNew, finalState, user, tiling);
    }

    __aicore__ inline void Process()
    {
        // 实际 kernel 在同一个 mix AIC/AIV launch 中按核类型进入对应 Process。
        if ASCEND_IS_AIC {
            cube_.Process();
        }
        if ASCEND_IS_AIV {
            vector_.Process();
        }
    }

private:
    ChunkGatedDeltaRuleFwdHCube<InputT, WorkspaceT, TileShapes, kGateMode> cube_{};
    ChunkGatedDeltaRuleFwdHVector<InputT, GateT, StateT, WorkspaceT, TileShapes, kGateMode, ExpMode> vector_{};
};

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          typename TileShapes, uint32_t kGateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHKernelImpl(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState,
    GM_ADDR tiling, GM_ADDR user)
{
    using FwdHKernel = ChunkGatedDeltaRuleFwdHKernel<InputT, GateT, StateT, WorkspaceT,
                                                      TileShapes, kGateMode, ExpMode>;
    FwdHKernel kernel;
    kernel.Init(k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState, user, tiling);
    kernel.Process();
}

template <typename InputT, typename GateT, typename TileShapes, uint32_t ExpMode>
__aicore__ inline void DispatchStateType(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState,
    GM_ADDR tiling, GM_ADDR user, int64_t stateDataType, bool useGk)
{
    using WorkspaceT = Fp32;
    if (stateDataType == 2) {
        if (useGk) {
            ChunkGatedDeltaRuleFwdHKernelImpl<InputT, GateT, Fp32, WorkspaceT, TileShapes, kGateGk, ExpMode>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState, tiling, user);
        } else {
            ChunkGatedDeltaRuleFwdHKernelImpl<InputT, GateT, Fp32, WorkspaceT, TileShapes, kGateG, ExpMode>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState, tiling, user);
        }
    } else {
        if (useGk) {
            ChunkGatedDeltaRuleFwdHKernelImpl<InputT, GateT, Bf16, WorkspaceT, TileShapes, kGateGk, ExpMode>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState, tiling, user);
        } else {
            ChunkGatedDeltaRuleFwdHKernelImpl<InputT, GateT, Bf16, WorkspaceT, TileShapes, kGateG, ExpMode>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState, tiling, user);
        }
    }
}

template <typename InputT, typename TileShapes, uint32_t ExpMode>
__aicore__ inline void DispatchGateType(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState,
    GM_ADDR tiling, GM_ADDR user, int64_t gDataType, int64_t stateDataType, bool useGk)
{
    if (gDataType == 2) {
        using GateT = typename FwdHGateTypeSelector<kGateG, Fp32>::type;
        DispatchStateType<InputT, GateT, TileShapes, ExpMode>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, user, stateDataType, useGk);
    } else {
        using GateT = typename FwdHGateTypeSelector<kGateG, Bf16>::type;
        DispatchStateType<InputT, GateT, TileShapes, ExpMode>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
            tiling, user, stateDataType, useGk);
    }
}

template <typename TileShapes>
__aicore__ inline void ChunkGatedDeltaRuleFwdHDispatch(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState,
    GM_ADDR tiling, GM_ADDR user)
{
    const auto deviceTiling = ReadKernelDispatchTiling(tiling);
    const uint32_t expMode = deviceTiling.useExp2 ? 1 : kExpModeNatural;
    if (deviceTiling.dataType == 1) {
        if (expMode == 1) {
            DispatchGateType<Bf16, TileShapes, 1>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, user, deviceTiling.gDataType, deviceTiling.stateDataType, deviceTiling.useGk);
        } else {
            DispatchGateType<Bf16, TileShapes, kExpModeNatural>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, user, deviceTiling.gDataType, deviceTiling.stateDataType, deviceTiling.useGk);
        }
    } else {
        if (expMode == 1) {
            DispatchGateType<Fp16, TileShapes, 1>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, user, deviceTiling.gDataType, deviceTiling.stateDataType, deviceTiling.useGk);
        } else {
            DispatchGateType<Fp16, TileShapes, kExpModeNatural>(
                k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState,
                tiling, user, deviceTiling.gDataType, deviceTiling.stateDataType, deviceTiling.useGk);
        }
    }
}

} // namespace GDN

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_h(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR h, GM_ADDR vNew, GM_ADDR finalState, GM_ADDR workspace, GM_ADDR tiling)
{
    // 与实际 kernel 一致：workspace 先转换为 user workspace，参数校验不在 device 入口执行。
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::ChunkGatedDeltaRuleFwdHDispatch<GDN::TileShapes128>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState, tiling, user);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::ChunkGatedDeltaRuleFwdHDispatch<GDN::TileShapes256>(
            k, w, u, g, gk, initialState, cuSeqlens, chunkIndices, h, vNew, finalState, tiling, user);
    }
}
