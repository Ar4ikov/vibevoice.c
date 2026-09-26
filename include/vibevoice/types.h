/**
 * @file types.h
 * @brief Core types, error codes, and constants for vibevoice.c
 */
#ifndef VV_TYPES_H
#define VV_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Status codes ──────────────────────────────────────────────────────── */

typedef enum vv_status {
    VV_OK = 0,

    /* General errors */
    VV_ERR_INVALID_ARG    = -1,
    VV_ERR_NULL_PTR       = -2,
    VV_ERR_OUT_OF_MEMORY  = -3,
    VV_ERR_NOT_FOUND      = -4,
    VV_ERR_IO             = -5,
    VV_ERR_PARSE          = -6,
    VV_ERR_UNSUPPORTED    = -7,
    VV_ERR_OVERFLOW       = -8,
    VV_ERR_SHAPE_MISMATCH = -9,
    /** A shared KV page pool had no pages left for a request that still fit
     *  its own window. Retryable: the pages come back as other requests end. */
    VV_ERR_KV_POOL_EXHAUSTED = -10,
    /** The caller asked for the work to stop (a streaming client left). */
    VV_ERR_CANCELLED      = -11,
    /** Every slot stayed taken for as long as the caller was willing to
     *  wait (a live session asking for one). Retryable. */
    VV_ERR_BUSY           = -12,

    /* CUDA errors */
    VV_ERR_CUDA           = -100,
    VV_ERR_CUDA_OOM       = -101,
    VV_ERR_CUDA_LAUNCH    = -102,

    /* TensorRT errors */
    VV_ERR_TRT            = -200,
    VV_ERR_TRT_BUILD      = -201,
    VV_ERR_TRT_RUNTIME    = -202,

    /* Model errors */
    VV_ERR_MODEL_FORMAT   = -300,
    VV_ERR_MODEL_VERSION  = -301,
    VV_ERR_WEIGHT_MISSING = -302,

    /* Audio errors */
    VV_ERR_AUDIO_FORMAT   = -400,
    VV_ERR_AUDIO_RESAMPLE = -401,
} vv_status_t;

/* ─── Data types ────────────────────────────────────────────────────────── */

typedef enum vv_dtype {
    VV_DTYPE_F32    = 0,
    VV_DTYPE_F16    = 1,
    VV_DTYPE_BF16   = 2,
    VV_DTYPE_U8     = 3,   /* uint8, used for NF4 packed data */
    VV_DTYPE_I32    = 4,
    VV_DTYPE_I64    = 5,
    VV_DTYPE_F8_E4M3 = 6,  /* FP8 for double-quantization scales */
    VV_DTYPE_NF4    = 7,   /* logical type: NF4 packed as U8 */
    VV_DTYPE_BOOL   = 8,
    VV_DTYPE_I8     = 9,   /* int8: W8A8 weights, int8 activations     */
    VV_DTYPE_I16    = 10,
    VV_DTYPE_U16    = 11,
    VV_DTYPE_U32    = 12,
    VV_DTYPE_F8_E5M2 = 13,
    /** A dtype string the parser does not know. Never guessed at: whoever
     *  needs the tensor refuses it by name instead of reading garbage. */
    VV_DTYPE_UNKNOWN = 14,
    VV_DTYPE_F64    = 15,
} vv_dtype_t;

/** @brief Size in bytes for a single element of the given dtype. */
static inline size_t vv_dtype_size(vv_dtype_t dtype) {
    switch (dtype) {
        case VV_DTYPE_F32:     return 4;
        case VV_DTYPE_F16:     return 2;
        case VV_DTYPE_BF16:    return 2;
        case VV_DTYPE_U8:      return 1;
        case VV_DTYPE_I32:     return 4;
        case VV_DTYPE_I64:     return 8;
        case VV_DTYPE_F8_E4M3: return 1;
        case VV_DTYPE_NF4:     return 1; /* packed: 2 values per byte */
        case VV_DTYPE_BOOL:    return 1;
        case VV_DTYPE_I8:      return 1;
        case VV_DTYPE_I16:     return 2;
        case VV_DTYPE_U16:     return 2;
        case VV_DTYPE_U32:     return 4;
        case VV_DTYPE_F8_E5M2: return 1;
        case VV_DTYPE_F64:     return 8;
        default:               return 0;
    }
}


/** @brief IEEE-754 half → float (scalar, host side). */
static inline float vv_half_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t man  = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 1;
            while (!(man & 0x400u)) { man <<= 1; exp--; }
            man &= 0x3FFu;
            bits = sign | ((exp + 127u - 15u) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (man << 13);
    }
    union { uint32_t u; float f; } c;
    c.u = bits;
    return c.f;
}

/**
 * @brief float → IEEE-754 half, rounding half away from zero (host side).
 *
 * An exact tie (1 + 2^-11, say) goes up, not to even. Kept as it is because
 * the AWQ repack's FP16 mins are computed with it and existing transcripts
 * depend on those bytes; new code converting weights wants
 * vv_float_to_half_rne().
 */
static inline uint16_t vv_float_to_half(float f) {
    union { float f; uint32_t u; } c;
    c.f = f;
    uint32_t sign = (c.u >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((c.u >> 23) & 0xFFu) - 127 + 15;
    uint32_t man  = c.u & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        int shift = 14 - exp;
        uint32_t round = (man >> (shift - 1)) & 1u;
        man >>= shift;
        man += round;
        return (uint16_t)(sign | man);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t round = (man >> 12) & 1u;
    man >>= 13;
    man += round;
    if (man & 0x400u) { man = 0; exp++; }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | man);
}

/**
 * @brief float → IEEE-754 half, round-to-nearest-even (host side).
 *
 * What PyTorch's `.half()` and the CUDA intrinsics do: an exact tie goes to
 * the even mantissa, so 1 + 2^-11 becomes 1.0 (0x3C00) and 1 + 3 * 2^-11
 * becomes 1 + 2^-9 (0x3C02). Overflow gives infinity, NaN stays NaN, and
 * values below half the smallest subnormal become signed zero.
 */
static inline uint16_t vv_float_to_half_rne(float f) {
    union { float f; uint32_t u; } c;
    c.f = f;
    const uint32_t sign = (c.u >> 16) & 0x8000u;
    const uint32_t absu = c.u & 0x7FFFFFFFu;
    if (absu >= 0x7F800000u)                       /* inf or NaN */
        return (uint16_t)(sign | 0x7C00u | (absu > 0x7F800000u ? 0x200u : 0u));
    if (absu >= 0x477FF000u)                       /* rounds past 65504 */
        return (uint16_t)(sign | 0x7C00u);
    if (absu < 0x38800000u) {                      /* FP16 subnormal or 0 */
        if (absu < 0x33000001u) return (uint16_t)sign;  /* <= 2^-25: to 0 */
        const uint32_t e = absu >> 23;             /* 102 .. 112 */
        const uint32_t m = (absu & 0x7FFFFFu) | 0x800000u;
        const uint32_t shift = 126u - e;           /* 14 .. 24 */
        uint32_t h = m >> shift;
        const uint32_t rest = m & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1u);
        if (rest > half || (rest == half && (h & 1u))) h++;
        return (uint16_t)(sign | h);
    }
    /* Normal: rebias the exponent, then round the 13 dropped bits. */
    uint32_t h = ((absu >> 13) - (112u << 10));
    const uint32_t rest = absu & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) h++;
    return (uint16_t)(sign | h);
}

/**
 * @brief bfloat16 → float. Exact: bfloat16 is the top half of a float, so
 *        every value, subnormals included, widens without rounding.
 */
static inline float vv_bf16_to_float(uint16_t b) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)b << 16;
    return c.f;
}

/* ─── Tensor descriptor ─────────────────────────────────────────────────── */

#define VV_MAX_DIMS 8

typedef struct vv_tensor {
    void*       data;               /**< Raw data pointer (CPU or GPU) */
    int64_t     shape[VV_MAX_DIMS]; /**< Shape dimensions */
    int         ndim;               /**< Number of dimensions */
    vv_dtype_t  dtype;              /**< Element data type */
    size_t      size_bytes;         /**< Total size in bytes */
    bool        on_gpu;             /**< true if data is on GPU */
} vv_tensor_t;

/** @brief Quantized tensor (NF4) with scale information. */
typedef struct vv_quant_tensor {
    vv_tensor_t packed;    /**< Packed uint8 data (2 NF4 values per byte) */
    vv_tensor_t scales;    /**< Per-block scales (FP16 or FP8) */
    int         block_size; /**< Quantization block size (typically 64) */
    bool        double_quant; /**< True if scales are also quantized */
    vv_tensor_t scale_scales; /**< Scales of scales (for double quant) */
    float       scale_offset; /**< Offset for double quantization */
} vv_quant_tensor_t;

/* ─── Model configuration (from config.json) ────────────────────────────── */

/**
 * @brief How the acoustic latent is drawn from the encoder's distribution.
 *
 * The encoder returns the mean of a Gaussian of fixed spread, not a latent.
 * `std_dist_type` in the checkpoint says which draw the reference takes; the
 * runtime defaults to the mode because that is reproducible.
 */
typedef enum vv_acoustic_sampling {
    VV_ACOUSTIC_MODE = 0,   /**< The mean. Deterministic; the default.      */
    VV_ACOUSTIC_FIX,        /**< mean + fix_std * N(0,1), elementwise.      */
    VV_ACOUSTIC_GAUSSIAN,   /**< Per-clip scale N(0, fix_std/0.8) times
                                 elementwise N(0,1) — this
                                 checkpoint's own setting.                  */
    VV_ACOUSTIC_SAMPLING_COUNT
} vv_acoustic_sampling_t;

typedef struct vv_acoustic_tokenizer_config {
    int   channels;
    bool  causal;
    int   vae_dim;
    float fix_std;
    /** What the checkpoint asks for; see vv_acoustic_sampling_t. */
    vv_acoustic_sampling_t std_dist_type;
    int   encoder_n_filters;
    int   encoder_ratios[8];
    int   n_ratios;
    int   encoder_depths[8];
    int   n_depths;
    float layernorm_eps;
    float layer_scale_init_value;
} vv_acoustic_tokenizer_config_t;

typedef struct vv_semantic_tokenizer_config {
    int   channels;
    bool  causal;
    int   vae_dim;
    int   encoder_n_filters;
    int   encoder_ratios[8];
    int   n_ratios;
    int   encoder_depths[8];
    int   n_depths;
    float layernorm_eps;
    float layer_scale_init_value;
} vv_semantic_tokenizer_config_t;

typedef struct vv_llm_config {
    int   hidden_size;
    int   num_hidden_layers;
    int   num_attention_heads;
    int   num_key_value_heads;
    int   head_dim;
    int   intermediate_size;
    int   vocab_size;
    int   max_position_embeddings;
    float rope_theta;
    float rms_norm_eps;
    /**
     * The LM head is the embedding table. Read at the root of config.json
     * or inside `decoder_config` (BitNet puts it there); the loader then
     * keeps one buffer for both and never reads a separate `lm_head.weight`.
     */
    bool  tie_word_embeddings;
    /** q/k/v carry a bias. Qwen2 always has one, so this defaults on. */
    bool  attention_bias;
} vv_llm_config_t;

/**
 * @brief Which published model a checkpoint is.
 *
 * All three share the speech front end and the Qwen2 layer; what differs is
 * the size, the prompt, the stop tokens and whether text comes out once per
 * clip or once per chunk. See family.h for what each one means at run time.
 */
typedef enum vv_model_family {
    VV_FAMILY_ASR_7B = 0,         /**< microsoft/VibeVoice-ASR and its 4-bit
                                       and AWQ derivatives                    */
    VV_FAMILY_ASR_BITNET,         /**< microsoft/VibeVoice-ASR-BitNet (1.5B)  */
    VV_FAMILY_ASR_STREAMING_7B,   /**< microsoft/VibeVoice-ASR-Streaming-7B   */
    VV_FAMILY_COUNT
} vv_model_family_t;

/* ─── Audio configuration ───────────────────────────────────────────────── */

/** @brief preprocessor_config.json, with the reference processor's defaults. */
typedef struct vv_audio_config {
    int   target_sample_rate;  /**< 24000 */
    bool  normalize_audio;     /**< true; false for the streaming model   */
    float target_db_fs;        /**< -25.0 */
    float eps;                 /**< 1e-6 */
    int   compress_ratio;      /**< 3200 */
    /** Streaming only: frames of text per chunk, 0 for one-shot models.  */
    int   chunk_frames;
    /** Streaming only: frames of audio past the chunk the model may see. */
    int   lookahead_frames;
} vv_audio_config_t;

typedef struct vv_model_config {
    vv_acoustic_tokenizer_config_t acoustic;
    vv_semantic_tokenizer_config_t semantic;
    vv_llm_config_t                llm;
    int   acoustic_vae_dim;
    int   semantic_vae_dim;
    vv_audio_config_t              audio;    /**< preprocessor_config.json */
    vv_model_family_t              family;   /**< see vv_config_load()     */
    char  architecture[64];                  /**< architectures[0], or ""  */
} vv_model_config_t;

/* ─── Inference initialization parameters ───────────────────────────────── */

/** @brief Weight placement strategy (auto-selected from VRAM budget). */
typedef enum vv_placement {
    VV_PLACE_ALL_GPU,       /**< All layers + embed + lm_head on GPU          */
    VV_PLACE_STREAM_FULL,   /**< Stream layers; embed + lm_head on GPU        */
    VV_PLACE_STREAM_EMBED,  /**< Stream layers + lm_head; embed on GPU only   */
    VV_PLACE_STREAM_ALL,    /**< Stream everything (layers + embed + lm_head) */
    VV_PLACE_CPU_ONLY,      /**< Everything on CPU, no CUDA at all            */
} vv_placement_t;

/* ─── Device selection ──────────────────────────────────────────────────── */

/** @brief Most devices one process will use. */
#define VV_MAX_GPUS 16

/**
 * @brief How much of a device this process may take.
 *
 * `bytes` when it is non-zero, otherwise `frac` of the device's *total*
 * memory. Total rather than free, because free moves under you the moment
 * anything else on the box starts, and "never take more than 8 GB" is the
 * thing an operator actually wants to say. What is free still caps the
 * result — a budget cannot spend memory another process is holding.
 */
typedef struct vv_mem_cap {
    size_t bytes;         /**< absolute cap; 0 = use `frac`                 */
    float  frac;          /**< fraction of the device total; 0 = uncapped   */
} vv_mem_cap_t;

/** @brief Which devices to use, in the order they should be filled. */
typedef struct vv_gpu_set {
    int          id[VV_MAX_GPUS];
    vv_mem_cap_t cap[VV_MAX_GPUS];
    int          n;       /**< 0 = nothing asked for; use the default device */
} vv_gpu_set_t;

/**
 * @brief What to do with more than one device.
 *
 * Replicas multiply throughput and shards do not: a replica shares nothing,
 * while a shard passes the hidden state along a chain and the devices take
 * turns. Shards buy capacity instead — the KV cache and the weights are
 * spread, so a window or a model that did not fit now does.
 */
typedef enum vv_split_mode {
    VV_SPLIT_AUTO = 0,  /**< replicate while each device holds a copy       */
    VV_SPLIT_REPLICA,   /**< a full copy on every device                    */
    VV_SPLIT_LAYER,     /**< one model, its layers spread over the devices  */
} vv_split_mode_t;

/** @brief Parse "auto" | "replica" | "layer"; count on failure. */
vv_split_mode_t vv_split_mode_parse(const char* name);

/** @brief Name for logging. */
const char* vv_split_mode_name(vv_split_mode_t mode);

#define VV_SPLIT_MODE_COUNT 3

/**
 * @brief Which attention kernels a context runs.
 *
 * All of them compute the same thing to within FP16 rounding; they differ in
 * how they use the card. `auto` picks per device and KV format (see
 * vv_attn_resolve in device.h) and is what every caller should leave it at
 * unless it is comparing kernels.
 */
typedef enum vv_attn_backend {
    VV_ATTN_AUTO = 0,     /**< fa2 where it runs (sm_75+), else fa1          */
    VV_ATTN_FA1,          /**< tiled scalar kernels, one query head at a
                               time, no tensor cores; the reference          */
    VV_ATTN_FA2,          /**< FlashAttention-2 prefill on tensor cores and a
                               GQA-grouped split-KV decode, bit-identical to
                               the kernels before the backends; pages on
                               FP16; sm_75+                                  */
    VV_ATTN_FLASHINFER,   /**< tensor-core prefill and decode, split KV for
                               few rows, any KV format, paged; fastest, and
                               within FP16 rounding of fa2; sm_75+           */
} vv_attn_backend_t;

#define VV_ATTN_BACKEND_COUNT 4

/** @brief Parse "auto" | "fa1" | "fa2" | "flashinfer" ("fi"); count on failure. */
vv_attn_backend_t vv_attn_backend_parse(const char* name);

/** @brief Name for logging. */
const char* vv_attn_backend_name(vv_attn_backend_t b);

/**
 * @brief Apply the environment to a requested backend.
 *
 * An explicit request wins. `auto` defers to VV_ATTN=fa1|fa2|flashinfer, and
 * VV_ATTN_MMA=0 (the older switch) still means fa1.
 */
vv_attn_backend_t vv_attn_backend_from_env(vv_attn_backend_t requested);

/**
 * @brief Whether the KV cache is one slab per context or pages from a pool.
 *
 * Paged caches take 64-position pages from one pool per device, so the slots
 * of a server share memory instead of each reserving a full window.
 */
typedef enum vv_kv_paging {
    VV_KV_PAGED_AUTO = 0, /**< paged when several slots share a device and
                               their kernels read pages as resolved          */
    VV_KV_PAGED_OFF,      /**< one contiguous slab per context               */
    VV_KV_PAGED_ON,       /**< pages from a pool shared by the slots          */
} vv_kv_paging_t;

/** @brief Parse "auto" | "on" | "off"; -1 on failure. */
int vv_kv_paging_parse(const char* name);

/**
 * @brief Where an asr-bitnet model is read from. Only BitNet has a choice:
 *        the GGUF pair VibeASR.cpp ships, or the F32 latent safetensors,
 *        ternarized and int8-quantized at load exactly as its converter does.
 */
typedef enum vv_weights_source {
    VV_SOURCE_AUTO = 0,     /**< the GGUF pair when present, else safetensors */
    VV_SOURCE_GGUF,
    VV_SOURCE_SAFETENSORS,
    VV_SOURCE_COUNT
} vv_weights_source_t;

/**
 * @brief How the speech encoder computes.
 *
 * FLOAT is the model as trained (FP32 on the CPU, FP16 on the GPU, exact
 * GELU). INT8 is VibeASR.cpp's encoder for BitNet: int8 weights and int8
 * activations with one scale per tensor, ReLU in the FFN (vae_i8.h), which
 * is what its transcripts come from. Only asr-bitnet has int8 weights.
 */
typedef enum vv_vae_numerics {
    VV_VAE_AUTO = 0,        /**< per family and backend, see pipeline.c     */
    VV_VAE_FLOAT,
    VV_VAE_INT8,
    VV_VAE_COUNT
} vv_vae_numerics_t;

/** @brief asr-bitnet's LM head: F16 as the reference computes it, or int8
 *         rows (half the bytes per decode step). */
typedef enum vv_head_format {
    VV_HEAD_AUTO = 0,       /**< F16                                        */
    VV_HEAD_F16,
    VV_HEAD_INT8,
    VV_HEAD_COUNT
} vv_head_format_t;

/**
 * @brief How a DFlash 2 drafter's projections are held on the device. A draft
 *        is only a proposal -- the target checks every token -- so the drafter
 *        may be quantized freely; what that costs is acceptance, not accuracy.
 */
typedef enum vv_drafter_quant {
    VV_DRAFTER_INT4 = 0,    /**< INT4 groups of 128 on the W4A16 kernels    */
    VV_DRAFTER_F16  = 1,    /**< as trained (BF16 -> FP16)                  */
    VV_DRAFTER_QUANT_COUNT
} vv_drafter_quant_t;

/** @brief "int4" | "f16"; VV_DRAFTER_QUANT_COUNT if unknown. */
vv_drafter_quant_t vv_drafter_quant_parse(const char* name);

/** @brief How a drafted block is checked (--draft-check). */
typedef enum vv_draft_check {
    VV_DRAFT_CHECK_AUTO  = 0,  /**< the default: exact                     */
    VV_DRAFT_CHECK_EXACT = 1,  /**< every row with its decode step's
                                    arithmetic: the transcript is byte for
                                    byte the one without a drafter        */
    VV_DRAFT_CHECK_FAST  = 2,  /**< the prefill kernels (tensor cores):
                                    cheaper per row, but a near-tie can go
                                    the other way than in a plain decode  */
    VV_DRAFT_CHECK_COUNT
} vv_draft_check_t;

/** @brief "auto" | "exact" | "fast"; VV_DRAFT_CHECK_COUNT if unknown. */
vv_draft_check_t vv_draft_check_parse(const char* name);

/** @brief "auto" | "gguf" | "safetensors"; VV_SOURCE_COUNT if unknown. */
vv_weights_source_t vv_weights_source_parse(const char* name);
const char* vv_weights_source_name(vv_weights_source_t s);
/** @brief "auto" | "float" | "int8"; VV_VAE_COUNT if unknown. */
vv_vae_numerics_t vv_vae_numerics_parse(const char* name);
const char* vv_vae_numerics_name(vv_vae_numerics_t v);
/** @brief "auto" | "f16" | "int8"; VV_HEAD_COUNT if unknown. */
vv_head_format_t vv_head_format_parse(const char* name);
const char* vv_head_format_name(vv_head_format_t h);

/** @brief Parameters for vv_inference_init(). Pass NULL for defaults. */
typedef struct vv_init_params {
    float  vram_budget;   /**< 0.0-1.0 fraction of free VRAM. Default: 1.0   */
    int    kv_format;     /**< vv_kv_format_t for the KV cache. 0 = FP16     */
    bool   cpu_only;      /**< Force CPU-only mode (vram_budget=0 shortcut)  */
    int    max_seq_len;   /**< KV-cache window in tokens. Default: 32768     */
    int    gpu_layers;    /**< Layers to keep on the GPU. -1 = fit to VRAM   */
    vv_gpu_set_t gpus;    /**< Devices and their caps. n = 0: use `gpu_id`   */
    int    split_mode;    /**< vv_split_mode_t across those devices          */
    int    weight_quant;  /**< vv_load_quant_t (quant.h). 0 = auto: keep the
                               checkpoint's own format                      */
    int    attn_backend;  /**< vv_attn_backend_t. Default: auto              */
    int    kv_paging;     /**< vv_kv_paging_t. Default: auto                 */
    /**
     * Contexts that will share this device's KV pool (the context itself
     * plus the clones made from it). Sizes a paged pool; 1 otherwise.
     */
    int    n_slots;
    /**
     * SmoothQuant statistics (smooth.h) folded into a dense checkpoint as it
     * is quantized at load (`weight_quant`), with their migration strength
     * and maps (0: the defaults, or VV_SMOOTH_ALPHA / VV_SMOOTH_MAPS).
     * NULL: none. Not owned; only read during vv_inference_init.
     */
    const struct vv_smooth_stats* smooth;
    float  smooth_alpha;
    int    smooth_maps;
    int    weights_source; /**< vv_weights_source_t (asr-bitnet). Default: auto */
    int    vae_numerics;   /**< vv_vae_numerics_t. Default: auto             */
    int    head_format;    /**< vv_head_format_t (asr-bitnet). Default: auto */
    /**
     * Directory of a DFlash 2 drafter for this model (spec.h): decode then
     * drafts a block of tokens in one pass and checks it with one pass of
     * the model. NULL: the model directory's own `drafter/` if it has one,
     * else plain decoding; "" (empty): plain decoding.
     */
    const char* draft_dir;
    int    draft_quant;   /**< vv_drafter_quant_t. Default: int4             */
    /**
     * Rows of a drafted block the model checks per pass: the last token and
     * draft_block - 1 drafts (2..the drafter's block size). The drafter
     * always drafts its whole block; checking a row costs about as much as
     * a step once there are more than two, so fewer rows can be faster when
     * the later drafts are rarely kept. 0: the default (4).
     */
    int    draft_block;
    int    draft_check;   /**< vv_draft_check_t. Default: auto (exact)      */
} vv_init_params_t;

/** @brief Fill vv_init_params_t with sane defaults. */
static inline vv_init_params_t vv_init_params_default(void) {
    /* Zeroed first: a field left out below (the per-device caps of an empty
     * `gpus`, say) must read as "not asked for", not as stack garbage. */
    vv_init_params_t p = {0};
    p.vram_budget = 1.0f;
    p.kv_format = 0;
    p.cpu_only = false;
    p.max_seq_len = 32768;
    p.gpu_layers = -1;
    p.gpus.n = 0;
    p.split_mode = VV_SPLIT_AUTO;
    p.weight_quant = 0;
    p.attn_backend = VV_ATTN_AUTO;
    p.kv_paging = VV_KV_PAGED_AUTO;
    p.n_slots = 1;
    p.smooth = NULL;
    p.smooth_alpha = 0.0f;
    p.smooth_maps = 0;
    p.weights_source = VV_SOURCE_AUTO;
    p.vae_numerics = VV_VAE_AUTO;
    p.head_format = VV_HEAD_AUTO;
    p.draft_dir = NULL;
    p.draft_quant = VV_DRAFTER_INT4;
    p.draft_block = 0;
    p.draft_check = VV_DRAFT_CHECK_AUTO;
    return p;
}

/* ─── Inference parameters ──────────────────────────────────────────────── */

typedef struct vv_inference_params {
    int          max_new_tokens;     /**< Default: 64000 */
    float        temperature;        /**< Default: 0.0 (greedy) */
    int          top_k;              /**< Default: 1 */
    const char** hotwords;           /**< NULL-terminated array */
    int          num_hotwords;
    bool         enable_timestamps;
    bool         enable_diarize;
    /** vv_acoustic_sampling_t. 0 = the mode, which is deterministic. */
    int          acoustic_sampling;
    /** Seed for that draw. Same seed, same latent, same transcript. */
    uint64_t     acoustic_seed;
} vv_inference_params_t;

/* ─── Transcription output ──────────────────────────────────────────────── */

typedef struct vv_segment {
    const char* speaker;
    float       start_time;
    float       end_time;
    const char* text;
} vv_segment_t;

typedef struct vv_transcription {
    vv_segment_t* segments;
    int           num_segments;
    const char*   full_text;
    float         duration;
    const char*   language;
    /**
     * The ids the model generated, every chunk's back to back (one chunk
     * for a one-shot model), without the tokens that stopped them. What a
     * tool that retraces a transcription needs (tools/dflash); may be NULL.
     */
    int32_t*      tokens;
    int           num_tokens;
    /** Per chunk: how many of `tokens` it produced, and the id that
     *  stopped it (-1 when the token cap did). */
    int*          chunk_tokens;
    int32_t*      chunk_stops;
    int           num_chunks;
} vv_transcription_t;

/* ─── Performance metrics ───────────────────────────────────────────────── */

typedef struct vv_perf_metrics {
    /* Timing — milliseconds */
    double total_ms;             /**< Wall-clock for entire transcribe()    */
    double audio_encode_ms;      /**< Step 1: Conv-VAE / feature prep       */
    double sequence_build_ms;    /**< Step 2: embed + upload                */
    double prefill_ms;           /**< Step 3: LLM prefill (28 layers)       */
    double decode_ms;            /**< Step 4: autoregressive decode loop    */
    double decode_layers_ms;     /**< ...of which: the 28 transformer layers */
    double decode_head_ms;       /**< ...of which: LM head + argmax + sync   */
    double decode_embed_ms;      /**< ...of which: token embedding lookup    */
    double postprocess_ms;       /**< Step 5: detokenize + JSON             */

    /* Key latencies */
    double ttft_ms;              /**< Time To First Token (steps 1-3)       */

    /* Throughput */
    double prefill_tok_per_sec;  /**< Prefill tokens / second               */
    double decode_tok_per_sec;   /**< Generated tokens / second             */

    /* Token counts */
    int    prefill_tokens;       /**< Tokens in the prefill sequence        */
    int    decode_tokens;        /**< Tokens generated during decode        */
    int    audio_frames;         /**< Audio feature frames (from encoder)   */

    /* Audio */
    float  audio_duration_sec;   /**< Input audio duration in seconds       */
    double rtf;                  /**< Real-Time Factor (lower = faster)     */

    /* GPU memory (bytes, 0 if unavailable) */
    size_t vram_used_bytes;      /**< VRAM in use after transcribe          */
    size_t vram_free_bytes;      /**< Free VRAM after transcribe            */
    size_t vram_total_bytes;     /**< Total GPU VRAM                        */

    /* KV cache */
    int    kv_cache_used;        /**< KV-cache positions filled             */
    int    kv_cache_max;         /**< KV-cache max capacity                 */
    float  kv_cache_pct;         /**< Utilization (0-100%)                  */

    /* Model info */
    int    num_layers;           /**< Transformer layers                    */
    int    hidden_size;          /**< Hidden dimension                      */
    int    kv_format;            /**< vv_kv_format_t of the KV cache        */
    size_t workspace_mb;         /**< Allocated workspace in MB             */
    int    attn_backend;         /**< vv_attn_backend_t the context ran     */
} vv_perf_metrics_t;

/* ─── Log levels ────────────────────────────────────────────────────────── */

typedef enum vv_log_level {
    VV_LOG_ERROR = 0,
    VV_LOG_WARN  = 1,
    VV_LOG_INFO  = 2,
    VV_LOG_DEBUG = 3,
} vv_log_level_t;

/* ─── NVTX helpers ──────────────────────────────────────────────────────── */

#ifdef VV_ENABLE_NVTX
#include <nvToolsExt.h>
#define VV_NVTX_PUSH(name) nvtxRangePushA(name)
#define VV_NVTX_POP()      nvtxRangePop()
#else
#define VV_NVTX_PUSH(name) ((void)0)
#define VV_NVTX_POP()      ((void)0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* VV_TYPES_H */
