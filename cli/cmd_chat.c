/**
 * @file cmd_chat.c
 * @brief `vv_cli chat` and `vv_cli mic` — the model stays loaded between clips.
 *
 * Both commands exist for the same reason: loading the 7B takes ~10 s, and
 * paying that per file makes short clips absurd. The engine comes up once and
 * then answers as fast as the audio allows.
 *
 * chat  a prompt loop: give it a path, or `rec 5` to record five seconds
 * mic   continuous capture, segmented by the VAD, transcribed as you speak
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/engine.h"
#include "vibevoice/capture.h"
#include "vibevoice/audio.h"
#include "vibevoice/kv_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <math.h>

static volatile int g_stop = 0;
static void on_signal(int sig) { (void)sig; g_stop = 1; }

typedef struct {
    vv_engine_params_t ep;
    const char* gpus;
    const char* gpu_memory;
    const char* device;
    const char* mic_file;      /* feed a WAV instead of a device */
    const char* hotwords;
    const char* save_dir;      /* write every captured utterance here */
    float start_db, stop_db, hangover_ms, max_segment_s;
    float gain_db, noise_margin_db;
    bool  no_adapt;
    bool  meter;
    bool  timestamps;
    bool  list_devices;
    bool  verbose;
} chat_args_t;

static void usage(const char* which) {
    if (strcmp(which, "mic") == 0) {
        fprintf(stderr,
            "Usage: vv_cli mic --model <dir> [options]\n\n"
            "  --device <sel>        Capture device: an index from\n"
            "                        --list-devices, its name, or part of one\n"
            "  --list-devices        Print the capture devices and exit\n"
            "  --from-file <wav>     Replay a WAV in real time instead of a mic\n"
            "  --start-db <dB>       Speech threshold (default: -38)\n"
            "  --stop-db <dB>        Silence threshold (default: -45)\n"
            "  --hangover <ms>       Silence before a segment closes (700)\n"
            "  --max-segment <s>     Force a cut after this long (30)\n"
            "  --gain <dB>           Amplify the captured signal\n"
            "  --meter               Show the live input level and the gate\n"
            "  --noise-margin <dB>   Speech must beat the noise floor by\n"
            "                        this much (default: 12)\n"
            "  --no-adapt            Take --start-db as-is, without tracking\n"
            "                        the noise floor\n"
            "  --save-audio <dir>    Write every captured utterance as WAV\n"
            "  --timestamps          Print segment start/end times\n");
    } else {
        fprintf(stderr,
            "Usage: vv_cli chat --model <dir> [options]\n\n"
            "Commands inside the prompt:\n"
            "  <path>                Transcribe an audio file\n"
            "  rec <seconds>         Record from the microphone, then transcribe\n"
            "  devices               List capture devices\n"
            "  device <sel>          Pick the one `rec` records from\n"
            "  hotwords a,b,c        Set hotwords for later requests\n"
            "  stats                 Engine counters\n"
            "  quit                  Exit\n");
    }
    fprintf(stderr,
        "\nCommon:\n"
        "  --model <dir>         Model directory (required)\n"
        "  --gpu <id>            GPU device id\n"
        "  --gpus <list>         Devices to spread slots over: 0,1 | all\n"
        "  --gpu-memory <size>   Cap per device: 80%% | 18GiB | 8192M | bytes\n"
        "  --split-mode <how>    auto (default) | replica | layer\n"
        "  --slots <n>           Concurrent transcriptions\n"
        "  --max-seq-len <n>     KV window\n"
        "  --kv-cache <fmt>      fp16 | fp8 | tq4 | ...\n"
        "  --quant <fmt>         auto (default) | none | nf4 | int4 | int8 |\n"
        "                        w8a8 | w4a8\n"
        "  --vae <numerics>      auto (default) | float | int8 (BitNet)\n"
        "  --source <from>       BitNet weights: auto | gguf | safetensors\n"
        "  --draft <dir>         DFlash 2 drafter for this model: decode drafts\n"
        "                        a block of tokens and checks it in one pass\n"
        "                        (tools/dflash; same tokens, fewer passes).\n"
        "                        Default: <model>/drafter if there is one;\n"
        "                        none: never\n"
        "  --draft-quant <fmt>   drafter weights: int4 (default) | f16\n"
        "  --draft-block <n>     rows of a block checked per pass (2..block,\n"
        "                        default 4)\n"
        "  --draft-check <how>   exact (default: the transcript is the one\n"
        "                        without --draft, byte for byte) | fast (the\n"
        "                        prefill kernels; a near-tie may go the\n"
        "                        other way)\n"
        "  --head <fmt>          BitNet CPU head: auto | f16 | int8\n"
        "  --attn <backend>      auto (default) | fa1 | fa2 | flashinfer —\n"
        "                        attention kernels; VV_ATTN= does the same\n"
        "  --kv-paged <mode>     auto | on | off\n"
        "  --hotwords a,b,c      Hotwords\n"
        "  --cpu                 CPU-only\n"
        "  --verbose\n");
}

/** @brief Turn --gpus/--gpu-memory into the engine's device set. */
static int resolve_gpus(chat_args_t* a) {
    return vv_gpu_set_from_flags(a->gpus, a->gpu_memory, a->ep.cpu_only,
                                 &a->ep.gpu_id, &a->ep.gpus) == VV_OK ? 0 : 1;
}

static int parse_common(int argc, char** argv, chat_args_t* a,
                        const char* which) {
    a->ep = vv_engine_params_default();
    a->start_db = -38.0f;
    a->stop_db = -45.0f;
    a->hangover_ms = 700.0f;
    a->max_segment_s = 30.0f;
    a->noise_margin_db = vv_vad_params_default().noise_margin_db;

    for (int i = 0; i < argc; i++) {
        const char* s = argv[i];
        const char* next = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (strcmp(s, "--model") == 0 && next) a->ep.model_dir = argv[++i];
        else if (strcmp(s, "--gpu") == 0 && next) a->ep.gpu_id = atoi(argv[++i]);
        else if (strcmp(s, "--gpus") == 0 && next) a->gpus = argv[++i];
        else if (strcmp(s, "--gpu-memory") == 0 && next) a->gpu_memory = argv[++i];
        else if (strcmp(s, "--split-mode") == 0 && next) {
            const vv_split_mode_t m = vv_split_mode_parse(argv[++i]);
            if (m >= VV_SPLIT_MODE_COUNT) {
                fprintf(stderr, "error: --split-mode is auto, replica or "
                                "layer, not '%s'\n", argv[i]);
                return -1;
            }
            a->ep.split_mode = (int)m;
        }
        else if (strcmp(s, "--slots") == 0 && next) a->ep.n_slots = atoi(argv[++i]);
        else if (strcmp(s, "--max-seq-len") == 0 && next) a->ep.max_seq_len = atoi(argv[++i]);
        else if (strcmp(s, "--gpu-layers") == 0 && next) a->ep.gpu_layers = atoi(argv[++i]);
        else if (strcmp(s, "--vram-budget") == 0 && next) a->ep.vram_budget = (float)atof(argv[++i]);
        else if (strcmp(s, "--attn") == 0 && next) {
            const vv_attn_backend_t b = vv_attn_backend_parse(argv[++i]);
            if (b >= VV_ATTN_BACKEND_COUNT) {
                fprintf(stderr, "error: --attn is auto, fa1, fa2 or "
                                "flashinfer, not '%s'\n", argv[i]);
                return -1;
            }
            a->ep.attn_backend = (int)b;
        }
        else if (strcmp(s, "--kv-paged") == 0 && next) {
            const int m = vv_kv_paging_parse(argv[++i]);
            if (m < 0) {
                fprintf(stderr, "error: --kv-paged is auto, on or off, not "
                                "'%s'\n", argv[i]);
                return -1;
            }
            a->ep.kv_paging = m;
        }
        else if (strcmp(s, "--kv-cache") == 0 && next) {
            vv_kv_format_t f = vv_kv_format_parse(argv[++i]);
            if (f >= VV_KV_FORMAT_COUNT) {
                fprintf(stderr, "error: unknown --kv-cache format '%s'\n", argv[i]);
                return -1;
            }
            a->ep.kv_format = (int)f;
        }
        else if (strcmp(s, "--quant") == 0 && next) {
            const vv_load_quant_t q = vv_load_quant_parse(argv[++i]);
            if (q >= VV_LOAD_QUANT_COUNT) {
                fprintf(stderr, "error: --quant is auto, none, nf4, int4, int8, w8a8 or w4a8, "
                                "not '%s'\n", argv[i]);
                return -1;
            }
            a->ep.weight_quant = (int)q;
        }
        else if (strcmp(s, "--vae") == 0 && next) {
            const vv_vae_numerics_t v = vv_vae_numerics_parse(argv[++i]);
            if (v >= VV_VAE_COUNT) {
                fprintf(stderr, "error: --vae is auto, float or int8, not '%s'\n",
                        argv[i]);
                return -1;
            }
            a->ep.vae_numerics = (int)v;
        }
        else if (strcmp(s, "--source") == 0 && next) {
            const vv_weights_source_t v = vv_weights_source_parse(argv[++i]);
            if (v >= VV_SOURCE_COUNT) {
                fprintf(stderr, "error: --source is auto, gguf or safetensors, "
                                "not '%s'\n", argv[i]);
                return -1;
            }
            a->ep.weights_source = (int)v;
        }
        else if (strcmp(s, "--draft") == 0 && next) {
            /* "none": an empty dir, which also skips <model>/drafter. */
            a->ep.draft_dir = strcmp(argv[++i], "none") == 0 ? "" : argv[i];
        }
        else if (strcmp(s, "--draft-quant") == 0 && next) {
            const vv_drafter_quant_t v = vv_drafter_quant_parse(argv[++i]);
            if (v >= VV_DRAFTER_QUANT_COUNT) {
                fprintf(stderr, "error: --draft-quant is int4 or f16, not "
                                "'%s'\n", argv[i]);
                return -1;
            }
            a->ep.draft_quant = (int)v;
        }
        else if (strcmp(s, "--draft-check") == 0 && next) {
            const vv_draft_check_t v = vv_draft_check_parse(argv[++i]);
            if (v >= VV_DRAFT_CHECK_COUNT) {
                fprintf(stderr, "error: --draft-check is exact or fast, not "
                                "'%s'\n", argv[i]);
                return -1;
            }
            a->ep.draft_check = (int)v;
        }
        else if (strcmp(s, "--draft-block") == 0 && next) {
            a->ep.draft_block = atoi(argv[++i]);
            if (a->ep.draft_block < 2) {
                fprintf(stderr, "error: --draft-block is 2 or more, not '%s'\n",
                        argv[i]);
                return -1;
            }
        }
        else if (strcmp(s, "--head") == 0 && next) {
            const vv_head_format_t v = vv_head_format_parse(argv[++i]);
            if (v >= VV_HEAD_COUNT) {
                fprintf(stderr, "error: --head is auto, f16 or int8, not '%s'\n",
                        argv[i]);
                return -1;
            }
            a->ep.head_format = (int)v;
        }
        else if (strcmp(s, "--device") == 0 && next) a->device = argv[++i];
        else if (strcmp(s, "--from-file") == 0 && next) a->mic_file = argv[++i];
        else if (strcmp(s, "--hotwords") == 0 && next) a->hotwords = argv[++i];
        else if (strcmp(s, "--start-db") == 0 && next) a->start_db = (float)atof(argv[++i]);
        else if (strcmp(s, "--stop-db") == 0 && next) a->stop_db = (float)atof(argv[++i]);
        else if (strcmp(s, "--hangover") == 0 && next) a->hangover_ms = (float)atof(argv[++i]);
        else if (strcmp(s, "--max-segment") == 0 && next) a->max_segment_s = (float)atof(argv[++i]);
        else if (strcmp(s, "--gain") == 0 && next) a->gain_db = (float)atof(argv[++i]);
        else if (strcmp(s, "--noise-margin") == 0 && next) a->noise_margin_db = (float)atof(argv[++i]);
        else if (strcmp(s, "--no-adapt") == 0) a->no_adapt = true;
        else if (strcmp(s, "--save-audio") == 0 && next) a->save_dir = argv[++i];
        else if (strcmp(s, "--meter") == 0) a->meter = true;
        else if (strcmp(s, "--list-devices") == 0) a->list_devices = true;
        else if (strcmp(s, "--timestamps") == 0) a->timestamps = true;
        else if (strcmp(s, "--cpu") == 0) a->ep.cpu_only = true;
        else if (strcmp(s, "--verbose") == 0) a->verbose = true;
        else if (strcmp(s, "--help") == 0 || strcmp(s, "-h") == 0) { usage(which); return 1; }
        else {
            fprintf(stderr, "%s: unknown option '%s'\n\n", which, s);
            usage(which);
            return -1;
        }
    }
    if (!a->ep.model_dir && !a->list_devices) {
        fprintf(stderr, "%s: --model is required\n\n", which);
        usage(which);
        return -1;
    }
    return 0;
}

/** @brief Fill inference params, pointing hotwords into `scratch`. */
static void build_params(vv_inference_params_t* p, const char* csv,
                         char* scratch, size_t scratch_n,
                         const char** list, int max_list) {
    memset(p, 0, sizeof(*p));
    p->max_new_tokens = 64000;
    p->temperature = 0.0f;
    p->top_k = 1;
    p->enable_timestamps = true;
    p->enable_diarize = true;
    p->num_hotwords = 0;

    if (!csv || !csv[0]) return;
    snprintf(scratch, scratch_n, "%s", csv);
    int n = 0;
    char* tok = strtok(scratch, ",");
    while (tok && n < max_list) {
        while (*tok == ' ') tok++;
        if (*tok) list[n++] = tok;
        tok = strtok(NULL, ",");
    }
    p->hotwords = n ? list : NULL;
    p->num_hotwords = n;
}

static void print_result(const vv_transcription_t* tr,
                         const vv_perf_metrics_t* perf, bool timestamps) {
    if (!tr) return;
    /*
     * Fall back to the flat text when the model's answer did not parse into
     * segments -- it still holds the transcription, and dropping it because
     * the JSON was malformed would be the wrong trade.
     */
    if (timestamps && tr->num_segments > 0) {
        for (int i = 0; i < tr->num_segments; i++) {
            const vv_segment_t* s = &tr->segments[i];
            printf("[%6.2f - %6.2f] %-10s %s\n", s->start_time, s->end_time,
                   s->speaker ? s->speaker : "", s->text ? s->text : "");
        }
    } else {
        printf("%s\n", tr->full_text ? tr->full_text : "");
    }
    if (perf)
        fprintf(stderr, "  (%.2fs audio, %.0f ms, RTF %.3f, %.0f tok/s)\n",
                tr->duration, perf->total_ms, perf->rtf,
                perf->decode_tok_per_sec);
    fflush(stdout);
}

/* ─── input level meter ──────────────────────────────────────────────────── */

/**
 * @brief Redraw the one-line level meter in place.
 *
 * Answers the question the transcript cannot: is anything reaching the
 * microphone at all? A segment that comes back as `[Noise]` looks the same
 * whether the speaker was too quiet, the wrong input was picked, or the room
 * really was noisy -- the meter tells those apart while you speak.
 */
static void meter_draw(float db, float threshold_db, bool speech) {
    const int width = 30;
    /* -60 dBFS is inaudible, 0 is clipping; map that range onto the bar. */
    float frac = (db + 60.0f) / 60.0f;
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 1.0f) frac = 1.0f;
    const int filled = (int)(frac * (float)width + 0.5f);

    char bar[64];
    for (int i = 0; i < width; i++) bar[i] = (i < filled) ? '#' : '.';
    bar[width] = '\0';
    /* The gate is what the level has to beat; seeing both explains silence. */
    fprintf(stderr, "\r  %6.1f dBFS [%s] gate %6.1f  %-7s", db, bar,
            threshold_db, speech ? "speech" : "");
    fflush(stderr);
}

/** @brief Wipe the meter line so a transcript starts on clean ground. */
static void meter_clear(bool on) {
    if (on) { fprintf(stderr, "\r%*s\r", 76, ""); fflush(stderr); }
}

/**
 * @brief Save one captured utterance, so it can be listened to afterwards.
 *
 * The pipeline normalizes to -25 dBFS before the model sees anything, so a
 * quiet recording is not itself the problem -- but a recording of the wrong
 * input sounds wrong, and that is only audible by playing it back.
 */
static void save_utterance(const char* dir, int index, const float* pcm,
                           int len) {
    if (!dir || !dir[0] || len <= 0) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/utt-%04d.wav", dir, index);
    if (vv_audio_save_wav(path, pcm, len, 24000) == VV_OK)
        fprintf(stderr, "  saved %s (%.2f s)\n", path, len / 24000.0f);
    else
        fprintf(stderr, "  cannot write %s\n", path);
}

/* ─── devices ────────────────────────────────────────────────────────────── */

/**
 * @brief Print the capture devices, with what to pass back as --device.
 *
 * The id is shown only when it differs from the friendly name: on Windows it
 * is the dshow alternative name, which stays unambiguous when two endpoints
 * share a label — a combined headset, say, whose mic and output carry one
 * name, or two identical headsets on the same machine.
 */
static int print_devices(void) {
    vv_mic_device_t* d = NULL;
    int n = 0;
    if (vv_mic_list_devices(&d, &n) != VV_OK) {
        fprintf(stderr, "no capture devices found.\n");
#if defined(_WIN32) || defined(__APPLE__)
        fprintf(stderr, "Enumeration goes through ffmpeg: check that ffmpeg "
                        "is on PATH and that microphone access is allowed.\n");
#else
        fprintf(stderr, "Enumeration needs libasound (ALSA). Without it, pass "
                        "an arecord device name to --device directly.\n");
#endif
        return 1;
    }
    printf("capture devices (pass the index, the name, or part of it "
           "to --device):\n");
    for (int i = 0; i < n; i++) {
        printf("  [%d] %s%s\n", i, d[i].name,
               d[i].is_default ? "   (used when --device is absent)" : "");
        if (strcmp(d[i].id, d[i].name) != 0)
            printf("      id: %s\n", d[i].id);
    }
    vv_free(d);
    return 0;
}

int vv_cmd_devices(int argc, char** argv) {
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr,
                    "Usage: vv_cli devices\n\n"
                    "Lists the microphones `mic` and `chat` can open.\n");
            return 0;
        }
    }
    vv_log_set_level(VV_LOG_WARN);
    return print_devices();
}

/* ─── mic, streaming model ──────────────────────────────────────────────── */

typedef struct {
    bool timestamps;
    bool at_line_start;
} mic_live_t;

static void mic_stream_event(void* user, const vv_stream_event_t* ev) {
    mic_live_t* m = (mic_live_t*)user;
    if (ev->type == VV_STREAM_EVENT_DELTA && !m->timestamps) {
        fwrite(ev->text, 1, ev->text_len, stdout);
        fflush(stdout);
    } else if (ev->type == VV_STREAM_EVENT_CHUNK && m->timestamps &&
               ev->text_len > 0) {
        /* One line per chunk; the model's inline turn breaks stay visible. */
        printf("[%7.2f - %7.2f] ", ev->audio_start, ev->audio_end);
        for (size_t i = 0; i < ev->text_len; i++)
            fputc(ev->text[i] == '\n' ? '|' : ev->text[i], stdout);
        fputc('\n', stdout);
        fflush(stdout);
    } else if (ev->type == VV_STREAM_EVENT_ERROR) {
        fprintf(stderr, "\nmic: stream failed in chunk %lld: %s\n",
                (long long)ev->chunk_index, vv_status_str(ev->status));
    }
}

/**
 * @brief Live transcription with a streaming model: capture blocks go
 * straight into one session, and text appears chunk by chunk (2.93 s of
 * audio plus 0.53 s of lookahead each). No VAD: the model is trained on the
 * continuous stream, silence included.
 */
static int mic_stream(vv_engine_t* engine, vv_mic_t* mic, const chat_args_t* a,
                      float gain) {
    /* "A,B" -> "A, B", the way every prompt joins hotwords. */
    char ctx_info[VV_HOTWORDS_MAX];
    vv_hotwords_join_csv(a->hotwords, ctx_info, sizeof(ctx_info));

    mic_live_t live;
    live.timestamps = a->timestamps;
    live.at_line_start = true;
    vv_stream_params_t sp;
    vv_stream_params_default(&sp);
    sp.context_info = ctx_info[0] ? ctx_info : NULL;
    sp.on_event = mic_stream_event;
    sp.user = &live;

    vv_engine_stream_t* es = NULL;
    vv_status_t s = vv_engine_stream_open(engine, &sp, 24000, &es);
    if (s != VV_OK) {
        fprintf(stderr, "mic: cannot open a streaming session: %s\n",
                vv_status_str(s));
        return 1;
    }

    float buf[4096];
    while (!g_stop && s == VV_OK) {
        const int got = vv_mic_read(mic, buf, (int)(sizeof(buf) / sizeof(buf[0])));
        if (got == 0) {
            if (vv_mic_eof(mic)) break;
            vv_msleep(5);
            continue;
        }
        if (gain != 1.0f)
            for (int i = 0; i < got; i++) buf[i] *= gain;
        s = vv_engine_stream_push(es, buf, (size_t)got);
    }
    /* Whatever arrived before the stop is still worth transcribing. */
    if (s == VV_OK) s = vv_engine_stream_finish(es);
    if (!a->timestamps) printf("\n");

    vv_stream_stats_t st;
    vv_stream_get_stats(vv_engine_stream_session(es), &st);
    if (st.chunks > 0)
        fprintf(stderr, "mic: %lld chunk(s), %lld tokens, %.0f ms per chunk "
                "(slowest %.0f), KV %lld/%lld\n", (long long)st.chunks,
                (long long)st.tokens,
                (st.prefill_ms + st.decode_ms) / (double)st.chunks,
                st.max_chunk_ms, (long long)st.kv_len,
                (long long)st.kv_capacity);
    vv_engine_stream_close(es);
    return s == VV_OK ? 0 : 1;
}

/* ─── mic ────────────────────────────────────────────────────────────────── */

int vv_cmd_mic(int argc, char** argv) {
    chat_args_t a;
    memset(&a, 0, sizeof(a));
    const int pr = parse_common(argc, argv, &a, "mic");
    if (pr != 0) return pr > 0 ? 0 : 1;

    vv_log_set_level(a.verbose ? VV_LOG_DEBUG : VV_LOG_WARN);

    if (a.list_devices) return print_devices();
    if (resolve_gpus(&a) != 0) return 1;

    vv_engine_t* engine = NULL;
    vv_status_t s = vv_engine_create(&a.ep, &engine);
    if (s != VV_OK) {
        fprintf(stderr, "mic: cannot start engine: %s\n", vv_status_str(s));
        return 1;
    }

    vv_mic_t* mic = NULL;
    if (a.mic_file)
        s = vv_mic_open_file(a.mic_file, 24000, true, &mic);
    else
        s = vv_mic_open(24000, a.device, &mic);
    if (s != VV_OK) {
        fprintf(stderr, "mic: cannot open capture: %s\n", vv_status_str(s));
        fprintf(stderr, "mic: run `vv_cli devices` to see what is available, "
                        "then pass one to --device\n");
        vv_engine_free(engine);
        return 1;
    }

    if (vv_engine_is_streaming(engine)) {
        signal(SIGINT, on_signal);
        const char* label = vv_mic_device_label(mic);
        fprintf(stderr, "listening to %s via %s, streaming (Ctrl-C to stop)\n",
                label && label[0] ? label : "the default device",
                vv_mic_backend(mic));
        const float g = (a.gain_db != 0.0f) ? powf(10.0f, a.gain_db / 20.0f)
                                            : 1.0f;
        const int rc = mic_stream(engine, mic, &a, g);
        if (vv_mic_captured(mic) == 0)
            fprintf(stderr, "mic: no audio arrived from the capture device.\n");
        vv_mic_close(mic);
        vv_engine_free(engine);
        return rc;
    }

    vv_vad_params_t vp = vv_vad_params_default();
    vp.sample_rate = 24000;
    vp.start_db = a.start_db;
    vp.stop_db = a.stop_db;
    vp.hangover_ms = a.hangover_ms;
    vp.max_segment_s = a.max_segment_s;
    vp.noise_margin_db = a.noise_margin_db;
    vp.adapt = !a.no_adapt;

    vv_vad_t* vad = NULL;
    if (vv_vad_create(&vp, &vad) != VV_OK) {
        vv_mic_close(mic);
        vv_engine_free(engine);
        return 1;
    }

    signal(SIGINT, on_signal);
    {
        const char* label = vv_mic_device_label(mic);
        if (label && label[0])
            fprintf(stderr, "listening to %s via %s (Ctrl-C to stop)\n",
                    label, vv_mic_backend(mic));
        else
            fprintf(stderr, "listening via %s (Ctrl-C to stop)\n",
                    vv_mic_backend(mic));
    }

    char hot_scratch[512];
    const char* hot_list[32];
    vv_inference_params_t ip;
    build_params(&ip, a.hotwords, hot_scratch, sizeof(hot_scratch),
                 hot_list, 32);

    const float gain = (a.gain_db != 0.0f) ? powf(10.0f, a.gain_db / 20.0f)
                                           : 1.0f;
    float buf[4096];
    int n_segments = 0;
    int since_meter = 0;
    while (!g_stop) {
        const int got = vv_mic_read(mic, buf, (int)(sizeof(buf) / sizeof(buf[0])));
        if (got == 0) {
            if (vv_mic_eof(mic)) break;
            vv_msleep(5);
            continue;
        }
        if (gain != 1.0f)
            for (int i = 0; i < got; i++) buf[i] *= gain;

        float* seg = NULL;
        int seg_len = 0;
        bool have = vv_vad_push(vad, buf, got, &seg, &seg_len);

        if (a.meter) {
            since_meter += got;
            if (since_meter >= 2400) {          /* ~10 redraws a second */
                since_meter = 0;
                meter_draw(vv_vad_level_db(vad), vv_vad_threshold_db(vad),
                           vv_vad_active(vad));
            }
        }

        while (have) {
            meter_clear(a.meter);
            save_utterance(a.save_dir, n_segments, seg, seg_len);
            vv_transcription_t* tr = NULL;
            vv_perf_metrics_t perf;
            memset(&perf, 0, sizeof(perf));
            if (vv_engine_transcribe(engine, seg, seg_len, 24000, &ip,
                                     &tr, &perf) == VV_OK) {
                n_segments++;
                print_result(tr, &perf, a.timestamps);
                vv_transcription_free(tr);
            }
            vv_free(seg);
            seg = NULL;
            have = vv_vad_drain(vad, &seg, &seg_len);
        }
    }
    meter_clear(a.meter);

    /* Whatever is still open at the end is still worth transcribing. */
    {
        float* seg = NULL;
        int seg_len = 0;
        if (vv_vad_flush(vad, &seg, &seg_len)) {
            save_utterance(a.save_dir, n_segments, seg, seg_len);
            vv_transcription_t* tr = NULL;
            vv_perf_metrics_t perf;
            memset(&perf, 0, sizeof(perf));
            if (vv_engine_transcribe(engine, seg, seg_len, 24000, &ip,
                                     &tr, &perf) == VV_OK) {
                n_segments++;
                print_result(tr, &perf, a.timestamps);
                vv_transcription_free(tr);
            }
            vv_free(seg);
        }
    }

    /*
     * A recorder that fails to start still hands back a healthy-looking pipe
     * that closes at once, so silence is the only symptom. Say so, instead of
     * reporting zero utterances and leaving the user to guess.
     */
    if (vv_mic_captured(mic) == 0)
        fprintf(stderr,
                "mic: no audio arrived from the capture device. Run "
                "`vv_cli devices` and pass one to --device; the recorder's "
                "own error, if any, is above.\n");

    const uint64_t over = vv_mic_overruns(mic);
    if (over)
        fprintf(stderr, "mic: dropped %llu samples (capture outran decode)\n",
                (unsigned long long)over);
    fprintf(stderr, "mic: %d utterance(s)\n", n_segments);

    vv_vad_free(vad);
    vv_mic_close(mic);
    vv_engine_free(engine);
    return 0;
}

/* ─── chat ───────────────────────────────────────────────────────────────── */

static void transcribe_path(vv_engine_t* engine, const char* path,
                            const vv_inference_params_t* ip, bool timestamps) {
    float* pcm = NULL;
    int n = 0, sr = 0;
    vv_status_t s = vv_audio_load_any(path, &pcm, &n, &sr);
    if (s != VV_OK) {
        printf("cannot read '%s': %s\n", path, vv_status_str(s));
        return;
    }
    vv_transcription_t* tr = NULL;
    vv_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));
    s = vv_engine_transcribe(engine, pcm, n, sr, ip, &tr, &perf);
    vv_free(pcm);
    if (s != VV_OK) {
        printf("transcription failed: %s\n", vv_status_str(s));
        return;
    }
    print_result(tr, &perf, timestamps);
    vv_transcription_free(tr);
}

static void record_and_transcribe(vv_engine_t* engine, const char* device,
                                  float seconds,
                                  const vv_inference_params_t* ip,
                                  bool timestamps) {
    vv_mic_t* mic = NULL;
    if (vv_mic_open(24000, device, &mic) != VV_OK) {
        printf("no microphone available\n");
        return;
    }
    const int want = (int)(seconds * 24000.0f);
    float* pcm = (float*)vv_alloc((size_t)want * sizeof(float));
    if (!pcm) { vv_mic_close(mic); return; }

    printf("recording %.1f s...\n", seconds);
    fflush(stdout);
    int have = 0;
    while (have < want) {
        const int got = vv_mic_read(mic, pcm + have, want - have);
        if (got == 0) vv_msleep(5);
        have += got;
    }
    vv_mic_close(mic);

    vv_transcription_t* tr = NULL;
    vv_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));
    if (vv_engine_transcribe(engine, pcm, have, 24000, ip, &tr, &perf) == VV_OK) {
        print_result(tr, &perf, timestamps);
        vv_transcription_free(tr);
    }
    vv_free(pcm);
}

int vv_cmd_chat(int argc, char** argv) {
    chat_args_t a;
    memset(&a, 0, sizeof(a));
    const int pr = parse_common(argc, argv, &a, "chat");
    if (pr != 0) return pr > 0 ? 0 : 1;

    vv_log_set_level(a.verbose ? VV_LOG_DEBUG : VV_LOG_WARN);

    if (a.list_devices) return print_devices();
    if (resolve_gpus(&a) != 0) return 1;

    vv_engine_t* engine = NULL;
    vv_status_t s = vv_engine_create(&a.ep, &engine);
    if (s != VV_OK) {
        fprintf(stderr, "chat: cannot start engine: %s\n", vv_status_str(s));
        return 1;
    }

    char hot_scratch[512];
    const char* hot_list[32];
    char hot_csv[512];
    snprintf(hot_csv, sizeof(hot_csv), "%s", a.hotwords ? a.hotwords : "");
    vv_inference_params_t ip;
    build_params(&ip, hot_csv, hot_scratch, sizeof(hot_scratch), hot_list, 32);

    printf("vibevoice.c %s — model ready, %d slot(s).\n"
           "Type a file path, `rec <seconds>`, `devices`, `device <sel>`, "
           "`hotwords a,b`, `stats`, or `quit`.\n",
           vv_version(), vv_engine_slots(engine));

    char line[1024];
    char device_sel[256];
    for (;;) {
        printf("\n> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;

        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        while (n && line[n - 1] == ' ') line[--n] = '\0';
        char* cmd = line;
        while (*cmd == ' ') cmd++;
        if (!*cmd) continue;

        if (strcmp(cmd, "quit") == 0 || strcmp(cmd, "exit") == 0 ||
            strcmp(cmd, "/quit") == 0) break;

        if (strcmp(cmd, "devices") == 0) {
            print_devices();
            continue;
        }
        if (strncmp(cmd, "device", 6) == 0 &&
            (cmd[6] == ' ' || cmd[6] == '\0')) {
            const char* rest = cmd + 6;
            while (*rest == ' ') rest++;
            if (*rest) {
                /* `line` is reused by the next prompt, so keep a copy. */
                snprintf(device_sel, sizeof(device_sel), "%s", rest);
                a.device = device_sel;
            }
            printf("device: %s\n", a.device ? a.device : "(default)");
            continue;
        }
        if (strcmp(cmd, "stats") == 0) {
            uint64_t done = 0;
            int busy = 0;
            vv_engine_stats(engine, &done, &busy);
            printf("slots %d, busy %d, completed %llu\n",
                   vv_engine_slots(engine), busy, (unsigned long long)done);
            continue;
        }
        if (strncmp(cmd, "hotwords", 8) == 0) {
            const char* rest = cmd + 8;
            while (*rest == ' ') rest++;
            snprintf(hot_csv, sizeof(hot_csv), "%s", rest);
            build_params(&ip, hot_csv, hot_scratch, sizeof(hot_scratch),
                         hot_list, 32);
            printf("hotwords: %d term(s)\n", ip.num_hotwords);
            continue;
        }
        if (strncmp(cmd, "rec", 3) == 0 && (cmd[3] == ' ' || cmd[3] == '\0')) {
            const float secs = (cmd[3] == ' ') ? (float)atof(cmd + 4) : 5.0f;
            record_and_transcribe(engine, a.device,
                                  secs > 0.5f ? secs : 5.0f, &ip, a.timestamps);
            continue;
        }

        transcribe_path(engine, cmd, &ip, a.timestamps);
    }

    vv_engine_free(engine);
    return 0;
}
