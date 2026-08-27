// 仅伪代码。FwdH 的 tiling key、输入输出和 round 数据结构。
// 本文件对应真实 op_kernel 中的 chunk_gated_delta_rule_fwd_h_tiling_key.h。

#pragma once

#include <array>
#include <cstdint>

namespace fwd_h_pseudocode {

constexpr int kBatchTokens = 64;
constexpr int kKeyDim = 128;
constexpr int kValueDim = 128;
constexpr int kMaxRoundHeads = 4;
constexpr int kAivCount = 2;
constexpr int kLocalSlotsPerAiv = 2;
constexpr int kMaxKeySlots = 4;

enum class GateMode { ScalarG, KeyWiseGk };
enum class StateType { Bf16, Fp32 };
enum class StateLayout { KV, VK };
enum class SlotState { Free, Loading, Ready, Reading };
enum class kg_payload { raw_k, prepared_kg };

struct Shape {
    int64_t d0 = 0;
    int64_t d1 = 0;
    int64_t d2 = 0;
    int64_t d3 = 0;
    int64_t d4 = 0;
};

struct TensorRef {
    const void* gm = nullptr;
    Shape shape{};
    StateType dtype = StateType::Bf16;
    bool present = false;
};

struct ApiInputs {
    TensorRef k;       // g-only：raw [B, HK, T, K]；gk-only：prepared kg [B, HV, T, K]
    TensorRef w;       // [B, HV, T, K]
    TensorRef u;       // [B, HV, T, V]
    TensorRef g;       // g-only [B, HV, T]，用 d3 == 0 表示
    TensorRef gk;      // gk-only [B, HV, T, K]
    TensorRef initialState;
    TensorRef finalState;
    bool outputFinalState = false;
    bool saveNewValue = true;
    bool useExp2 = false;
    bool stateVFirst = false;
    int64_t chunkSize = kBatchTokens;
    const int64_t* cuSeqlens = nullptr;
    int64_t cuSeqlensLength = 0;
    const int64_t* chunkIndices = nullptr;
    int64_t chunkIndicesLength = 0;
};

struct ApiOutputs {
    TensorRef h;          // [B, HV, Ctot, K, V] 或 [B, HV, Ctot, V, K]
    TensorRef v_new;      // [B, HV, T, V]
    TensorRef finalState; // [N, HV, K, V] 或 [N, HV, V, K]
};

// 仅供 fast-launch 形状伪代码入口使用的框架侧名称。
using PseudocodeTensor = TensorRef;
using OptionalPseudocodeTensor = TensorRef;

struct OptionalIntArray {
    const int64_t* data = nullptr;
    int64_t size = 0;
};

struct PseudocodeTensorTuple {
    ApiOutputs outputs{};
    bool parameterError = false;
};

struct SequenceSpan {
    int sequence = 0;
    int64_t tokenBegin = 0;
    int64_t tokenEnd = 0;
    int64_t length = 0;
    int chunkCount = 0;
    int chunkPrefix = 0;
};

struct ChunkSpan {
    int sequence = 0;
    int chunk = 0;
    int globalChunk = 0;
    int64_t tokenBegin = 0;
    int64_t validTokens = 0; // M；经过 Host 校验后始终满足 1 <= M <= 64
    bool first = false;
    bool last = false;
};

struct HeadBinding {
    int roundHead = -1; // 0..activeHeadCount-1，仅在本 round 内有效
    int hv = -1;
    int kh = -1;        // g-only：hk；gk-only：hv
    int kgSlot = -1;    // 当前 round 的 kg slot 表中的 slot
    int hSlot = -1;     // L1[128,256)，每个 value head 一个 resident
    int wSlot = -1;     // L1[0,64)，每个 value head 一个 W
    int aiv = -1;       // roundHead % 2
    int localSlot = -1; // roundHead / 2
    bool active = false;
};

struct kg_binding {
    int slot = -1;
    int kh = -1;
    int firstConsumerRoundHead = -1;
    int lastConsumerRoundHead = -1;
    kg_payload payload = kg_payload::raw_k;
    SlotState state = SlotState::Free;
    uint64_t generation = 0;
};

struct RoundPlan {
    int sequence = -1;
    int round = -1;
    ChunkSpan chunk{};
    GateMode gateMode = GateMode::ScalarG;
    kg_payload kg_payload_kind = kg_payload::raw_k;
    int activeHvBegin = 0;
    int activeHvCount = 0;
    int requiredKhCount = 0;
    std::array<int, kMaxRoundHeads> requiredKh{};
    std::array<HeadBinding, kMaxRoundHeads> heads{};
    std::array<kg_binding, kMaxKeySlots> kg{};
    // 这三个分支依赖当前 sequence 的 chunk.first/last，只能由 BuildChunkPlan 填充。
    bool stage0Required = false;
    bool stage2Required = false;
    bool stage3Required = false;
};

struct TilingPlan {
    int64_t batch = 0;
    int64_t sequenceCount = 0;
    int64_t seqlen = 0;
    int64_t hk = 0;
    int64_t hv = 0;
    int64_t keyDim = kKeyDim;
    int64_t valueDim = kValueDim;
    int64_t chunkSize = kBatchTokens;
    StateType stateType = StateType::Fp32;
    GateMode gateMode = GateMode::ScalarG;
    StateLayout stateLayout = StateLayout::KV;
    bool varlen = false;
    bool outputFinalState = false;
    bool useInitialState = false;
    bool useExp2 = false;
    int totalChunks = 0;
    std::array<SequenceSpan, 64> sequences{};
};

struct HostResult {
    bool ok = false;
    TilingPlan tiling{};
    ApiOutputs outputs{};
    const char* error = nullptr;
};

} // 命名空间 fwd_h_pseudocode
