// 仅伪代码。FwdH 的 tiling key、输入输出和 round 数据结构。
// 本文件对应真实 op_kernel 中的 chunk_gated_delta_rule_fwd_h_tiling_key.h。

#ifndef FLA_FWD_H_PSEUDOCODE_CHUNK_GATED_DELTA_RULE_FWD_H_TILING_KEY_H_
#define FLA_FWD_H_PSEUDOCODE_CHUNK_GATED_DELTA_RULE_FWD_H_TILING_KEY_H_

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
enum class LocalDataOwner { Free, P, VNewWork, D, H0, HWrite };
enum class StateOwner { Free, InitialMte2, RollingMte2, RResident, RNextMte3 };

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

struct WorkspaceRefs {
    // 架构扩展 scratch；其含义只在对应 arch Cube 头文件解释，arch35 不访问这两组槽位。
    std::array<TensorRef, kMaxRoundHeads> cubeScratch0{};
    std::array<TensorRef, kMaxRoundHeads> cubeScratch1{};
    // 右操作数的私有 GM scratch。Stage1 写入 ND，Stage2 再搬成 L1 NZ；
    // 每个 head_round slot 只保存当前 chunk，GM 搬入 L1 后立即归还。
    TensorRef rightOperandGm;
    // 以下 offset 由 kernel tiling 直接给出，所有地址都相对 GetUserWorkspace(workspace)。
    const void* userWorkspace = nullptr;
    int64_t vWorkspaceOffset = 0;
    int64_t vUpdateWorkspaceOffset = 0;
    int64_t kDecayWorkspaceOffset = 0;
    int64_t hWorkspaceOffset = 0;
    int64_t numSeqWorkspaceOffset = 0;
    int64_t numChunksWorkspaceOffset = 0;
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

struct LocalDataTicket {
    int bank = -1;
    LocalDataOwner owner = LocalDataOwner::Free;
    LocalDataOwner previousOwner = LocalDataOwner::Free;
    uint64_t generation = 0;
};

struct StateTicket {
    int bank = -1;
    StateOwner owner = StateOwner::Free;
    uint64_t generation = 0;
};

struct RoundPlan {
    int sequence = -1;
    int round = -1;
    ChunkSpan chunk{};
    GateMode gateMode = GateMode::ScalarG;
    StateType stateType = StateType::Fp32;
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
    bool finalVNewOnly = false;
    bool hasNextChunk = false;
    bool hasNextHeadRound = false;
    bool nextRoundStartsWithS0 = false;
    bool nextRoundStartsWithS1NoP = false;
    bool roundBoundaryDrained = false;
};

struct TilingPlan {
    int64_t batch = 0;
    int64_t shapeBatch = 0; // dense 为真实 B；varlen 固定为 1
    int64_t tokenBatch = 0; // dense 为 1；varlen 为 sequence 数
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
    bool useG = false;
    bool useGk = false;
    bool useExp2 = false;
    bool stateVFirst = false;
    int totalChunks = 0;
    std::array<SequenceSpan, 64> sequences{};
};

// 设备入口从 tiling GM 地址读取的原始数据。字段顺序必须与 op_host 的
// BEGIN_TILING_DATA_DEF(ChunkGatedDeltaRuleFwdHTilingData) 完全一致。
//
// batch/seqlen/head/dim/chunkSize：形状和分块参数；
// useInitialState/storeFinalState：S-1、S0、S3 的首尾分支；
// dataType/gDataType/stateDataType：入口模板分发，不能在 VF 内运行期判断；
// isVariedLen/shapeBatch/tokenBatch：dense/varlen 的 sequence 解释；
// useG/useGk：g-only 或 gk-only，以及 required_hk_round 的映射模式；
// useExp2/stateVFirst：设计文档要求透传到 kernel 的属性；
// 六个 workspace offset：GM workspace 子区的起始字节偏移。
//
// 当前仓库的真实 op_host tiling 还没有同时写入 useExp2/stateVFirst；落地时必须在
// host/kernel 两侧以相同顺序补齐，不能在 kernel 中猜默认值或依赖外部转置。
struct KernelTilingData {
    int64_t batch = 0;
    int64_t seqlen = 0;
    int64_t kNumHead = 0;
    int64_t vNumHead = 0;
    int64_t kHeadDim = 0;
    int64_t vHeadDim = 0;
    int64_t chunkSize = kBatchTokens;
    bool useInitialState = false;
    bool storeFinalState = false;
    int64_t dataType = 1;
    int64_t gDataType = 1;
    int64_t stateDataType = 2;
    int64_t isVariedLen = 0;
    int64_t shapeBatch = 0;
    int64_t tokenBatch = 0;
    bool useG = false;
    bool useGk = false;
    bool useExp2 = false;
    bool stateVFirst = false;
    int64_t vWorkspaceOffset = 0;
    int64_t vUpdateWorkspaceOffset = 0;
    int64_t kDecayWorkspaceOffset = 0;
    int64_t hWorkspaceOffset = 0;
    int64_t numSeqWorkspaceOffset = 0;
    int64_t numChunksWorkspaceOffset = 0;
};

struct HostResult {
    bool ok = false;
    TilingPlan tiling{};
    ApiOutputs outputs{};
    WorkspaceRefs workspace{};
    const char* error = nullptr;
};

} // 命名空间 fwd_h_pseudocode

#endif // FLA_FWD_H_PSEUDOCODE_CHUNK_GATED_DELTA_RULE_FWD_H_TILING_KEY_H_
