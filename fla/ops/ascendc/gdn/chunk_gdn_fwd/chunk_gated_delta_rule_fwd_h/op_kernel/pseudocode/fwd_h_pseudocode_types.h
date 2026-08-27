// PSEUDOCODE ONLY. This file is an implementation blueprint and is not compiled.
// The directory mirrors the PR370 FwdH layout; no PR370 kernel logic is reused.

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
    TensorRef k;       // g-only: raw [B, HK, T, K]; gk-only: prepared kg [B, HV, T, K]
    TensorRef w;       // [B, HV, T, K]
    TensorRef u;       // [B, HV, T, V]
    TensorRef g;       // g-only [B, HV, T], represented by d3 == 0
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
    TensorRef h;          // [B, HV, Ctot, K, V] or [B, HV, Ctot, V, K]
    TensorRef v_new;      // [B, HV, T, V]
    TensorRef finalState; // [N, HV, K, V] or [N, HV, V, K]
};

// Framework-facing names used only by the fast-launch-shaped pseudocode entry.
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
    int64_t validTokens = 0; // M, always 1 <= M <= 64 after host validation
    bool first = false;
    bool last = false;
};

struct HeadBinding {
    int roundHead = -1; // 0..activeHeadCount-1, local to this round
    int hv = -1;
    int kh = -1;        // g-only: hk; gk-only: hv
    int kgSlot = -1;    // slot in the current round's kg slot table
    int hSlot = -1;     // L1[128,256), one resident per value head
    int wSlot = -1;     // L1[0,64), one W per value head
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
    bool stage0Required = true;
    bool stage2Required = true;
    bool stage3Required = true;
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

struct L1SlotTable {
    // Address units are KiB and are fixed for the whole dispatch.
    int wBaseKiB = 0;       // [0, 64): 4 x 16 KiB W slots
    int hBaseKiB = 128;     // [128, 256): 4 x 32 KiB H slots
    int kgBaseKiB = 256;    // [256, 320): 4 x 16 KiB kg slots
    int rightBaseKiB = 128; // S1 reuses an H slot only after its S0 read is free
};

struct UbSlotTable {
    int localDataBaseKiB[kLocalSlotsPerAiv] = {0, 64};
    int pBaseKiB[kLocalSlotsPerAiv] = {0, 64};
    int dBaseKiB[kLocalSlotsPerAiv] = {0, 64};
    int v_new_work_base_kib[kLocalSlotsPerAiv] = {128, 160};
    int stateScratchBaseKiB = 160;
    int gateBaseKiB = 224;
};

struct EventToken {
    int id = -1;
    uint64_t generation = 0;
    bool valid = false;
};

struct SlotEvents {
    EventToken ready{};
    EventToken free{};
    EventToken terminalDrain{};
};

inline int64_t state_gm_offset(int64_t base, int64_t k, int64_t v, bool stateVFirst)
{
    // Internal L1/UB is always canonical [K,V]; only GM state order changes.
    return base + (stateVFirst ? v * kValueDim + k : k * kValueDim + v);
}

} // namespace fwd_h_pseudocode
