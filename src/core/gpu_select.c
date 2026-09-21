/**
 * @file gpu_select.c
 * @brief Which devices to use, and how much of each.
 *
 * Two knobs, both of which have to be readable by whoever is on call:
 * `--gpus 0,2` picks devices, `--gpu-memory 20G,80%` says how much of each
 * may be spent. Both are parsed here rather than in the CLI so the server,
 * the chat loop and the library entry point all agree on what "80%" means.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/device.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <dirent.h>
#include <pthread.h>
#endif

/**
 * @brief Parse one memory cap: "8G", "18GiB", "8192M", "80%", "8589934592".
 *
 * Both conventions are accepted for the suffixes and both mean 1024: a card
 * advertised as 24 GB has 24 GiB, everyone says "24 gig", and a runtime that
 * quietly read that as 24 * 10^9 would hand back 2% less memory than asked
 * for with no way to tell.
 */
vv_status_t vv_parse_mem_cap(const char* text, vv_mem_cap_t* out) {
    if (!text || !out) return VV_ERR_NULL_PTR;
    while (isspace((unsigned char)*text)) text++;

    char* end = NULL;
    const double v = strtod(text, &end);
    if (end == text || v < 0.0) return VV_ERR_INVALID_ARG;
    while (isspace((unsigned char)*end)) end++;

    out->bytes = 0;
    out->frac = 0.0f;

    if (*end == '%') {
        if (v > 100.0) return VV_ERR_INVALID_ARG;
        out->frac = (float)(v / 100.0);
        end++;
    } else {
        double mul = 1.0;
        switch (toupper((unsigned char)*end)) {
            case 'K': mul = 1024.0; end++; break;
            case 'M': mul = 1024.0 * 1024.0; end++; break;
            case 'G': mul = 1024.0 * 1024.0 * 1024.0; end++; break;
            case 'T': mul = 1024.0 * 1024.0 * 1024.0 * 1024.0; end++; break;
            case 'B': case '\0': break;
            default: return VV_ERR_INVALID_ARG;
        }
        /* Swallow the rest of "GiB" / "GB" / "B". */
        if (toupper((unsigned char)*end) == 'I') end++;
        if (toupper((unsigned char)*end) == 'B') end++;
        out->bytes = (size_t)(v * mul);
        if (out->bytes == 0) return VV_ERR_INVALID_ARG;
    }

    while (isspace((unsigned char)*end)) end++;
    return *end == '\0' ? VV_OK : VV_ERR_INVALID_ARG;
}

size_t vv_mem_cap_bytes(const vv_mem_cap_t* cap, size_t total) {
    if (!cap) return 0;
    if (cap->bytes) return cap->bytes;
    if (cap->frac > 0.0f) return (size_t)((double)total * (double)cap->frac);
    return 0;   /* uncapped */
}

/**
 * @brief Parse `--gpus`: "0,1", "all", or a single id.
 *
 * Leaves every cap uncapped; vv_gpu_set_caps fills those in separately so
 * that the two flags can be given in either order.
 */
vv_status_t vv_gpu_set_parse(const char* text, vv_gpu_set_t* out) {
    if (!text || !out) return VV_ERR_NULL_PTR;
    memset(out, 0, sizeof(*out));

    if (strcmp(text, "all") == 0) {
        /* Counting devices is the first thing that starts CUDA. */
        vv_cuda_fix_visible_devices();
        const int n = vv_dev_device_count();
        if (n <= 0) {
            VV_LOG_E("gpus: 'all' asked for, but no device is visible");
            return VV_ERR_NOT_FOUND;
        }
        out->n = n > VV_MAX_GPUS ? VV_MAX_GPUS : n;
        for (int i = 0; i < out->n; i++) out->id[i] = i;
        return VV_OK;
    }

    const char* p = text;
    while (*p) {
        while (*p == ',' || isspace((unsigned char)*p)) p++;
        if (!*p) break;
        char* end = NULL;
        const long v = strtol(p, &end, 10);
        if (end == p || v < 0) {
            VV_LOG_E("gpus: '%s' is not a device list", text);
            return VV_ERR_INVALID_ARG;
        }
        for (int i = 0; i < out->n; i++)
            if (out->id[i] == (int)v) {
                VV_LOG_E("gpus: device %ld listed twice", v);
                return VV_ERR_INVALID_ARG;
            }
        if (out->n >= VV_MAX_GPUS) {
            VV_LOG_E("gpus: at most %d devices", VV_MAX_GPUS);
            return VV_ERR_INVALID_ARG;
        }
        out->id[out->n++] = (int)v;
        p = end;
        while (isspace((unsigned char)*p)) p++;
        if (*p && *p != ',') {
            VV_LOG_E("gpus: '%s' is not a device list", text);
            return VV_ERR_INVALID_ARG;
        }
    }
    if (out->n == 0) {
        VV_LOG_E("gpus: empty device list");
        return VV_ERR_INVALID_ARG;
    }
    return VV_OK;
}

/**
 * @brief Parse `--gpu-memory`: one cap for every device, or one each.
 *
 * A single value applies to all of them, which is what somebody typing
 * `--gpu-memory 80%` on a two-card box means. A list must match the device
 * list exactly rather than being padded, because a silently uncapped second
 * card is precisely the surprise this flag exists to prevent.
 */
vv_status_t vv_gpu_set_caps(const char* text, vv_gpu_set_t* set) {
    if (!text || !set) return VV_ERR_NULL_PTR;

    vv_mem_cap_t caps[VV_MAX_GPUS];
    int n = 0;
    const char* p = text;
    while (*p && n < VV_MAX_GPUS) {
        const char* comma = strchr(p, ',');
        char item[64];
        const size_t len = comma ? (size_t)(comma - p) : strlen(p);
        if (len == 0 || len >= sizeof(item)) {
            VV_LOG_E("gpu-memory: '%s' is not a size", text);
            return VV_ERR_INVALID_ARG;
        }
        memcpy(item, p, len);
        item[len] = '\0';
        const vv_status_t s = vv_parse_mem_cap(item, &caps[n]);
        if (s != VV_OK) {
            VV_LOG_E("gpu-memory: cannot read '%s' — expected a size like "
                     "18GiB, 8192M, 80%% or a byte count", item);
            return s;
        }
        n++;
        if (!comma) break;
        p = comma + 1;
    }
    if (n == 0) return VV_ERR_INVALID_ARG;

    const int devices = set->n > 0 ? set->n : 1;
    if (n == 1) {
        for (int i = 0; i < VV_MAX_GPUS; i++) set->cap[i] = caps[0];
        return VV_OK;
    }
    if (n != devices) {
        VV_LOG_E("gpu-memory: %d values for %d device(s) — give one value for "
                 "all of them, or one per device", n, devices);
        return VV_ERR_INVALID_ARG;
    }
    for (int i = 0; i < n; i++) set->cap[i] = caps[i];
    return VV_OK;
}

/**
 * @brief Settle the set against the machine and say what was chosen.
 *
 * An empty set becomes the single device the caller named, so every path
 * downstream can read the same structure whether or not --gpus was given.
 */
vv_status_t vv_gpu_set_resolve(vv_gpu_set_t* set, int fallback_id,
                               bool cpu_only) {
    if (!set) return VV_ERR_NULL_PTR;
    if (set->n == 0) {
        set->n = 1;
        set->id[0] = fallback_id;
    }
    if (cpu_only) return VV_OK;

    vv_cuda_fix_visible_devices();
    const int visible = vv_dev_device_count();
    if (visible <= 0) {
        /* No driver: the caller falls back to the CPU path and says so. */
        return VV_OK;
    }
    for (int i = 0; i < set->n; i++) {
        if (set->id[i] >= visible) {
            VV_LOG_E("gpus: device %d asked for, but only %d %s visible "
                     "(0..%d)", set->id[i], visible,
                     visible == 1 ? "is" : "are", visible - 1);
            /*
             * The ids count what this process can see, which is what
             * CUDA_VISIBLE_DEVICES or a container's device set left of the
             * machine — so a list copied from `nvidia-smi` names cards that
             * are not here.
             */
            VV_LOG_E("gpus: the ids count the cards this process can see, not "
                     "the host's; with CUDA_VISIBLE_DEVICES set, or under an "
                     "orchestrator, they start at 0 again. Use --gpus all");
            return VV_ERR_NOT_FOUND;
        }
    }

    for (int i = 0; i < set->n; i++) {
        char name[128] = "unknown";
        size_t total = 0, freem = 0;
        vv_dev_get_device_name(set->id[i], name, sizeof(name));
        vv_dev_get_device_info(set->id[i], &total, &freem, NULL);
        const size_t cap = vv_mem_cap_bytes(&set->cap[i], total);
        if (cap == 0) {
            VV_LOG_I("gpu %d: %s, %.1f/%.1f GiB free, no cap",
                     set->id[i], name, (double)freem / (1024.0*1024.0*1024.0),
                     (double)total / (1024.0*1024.0*1024.0));
        } else {
            VV_LOG_I("gpu %d: %s, %.1f/%.1f GiB free, capped at %.1f GiB",
                     set->id[i], name, (double)freem / (1024.0*1024.0*1024.0),
                     (double)total / (1024.0*1024.0*1024.0),
                     (double)cap / (1024.0*1024.0*1024.0));
            if (cap > freem)
                VV_LOG_W("gpu %d: the cap is above what is free; %.1f GiB is "
                         "the real limit", set->id[i],
                         (double)freem / (1024.0*1024.0*1024.0));
        }
    }
    return VV_OK;
}

/**
 * @brief Translate `CUDA_VISIBLE_DEVICES` into the numbering a container uses.
 *
 * A container is given a subset of the host's cards by mounting only their
 * device nodes, and CUDA then numbers what it finds from 0. An orchestrator
 * that also sets `CUDA_VISIBLE_DEVICES` to the *host* indexes — GPUStack
 * 2.2.2 does — hands the process a list from the other namespace. Given host
 * card 1 alone, the container holds `/dev/nvidia1`, the variable says `1`,
 * CUDA is asked for the second of one card and reports **none**: the model
 * lands on the CPU, or the run dies on `--gpus all`.
 *
 * The host's own numbering is the case where the two agree, so mapping each
 * listed number to its position among the nodes present is a no-op there and
 * the correction only where it is wrong. A list naming nothing present cannot
 * be honoured at all — that is `VV_ERR_NOT_FOUND`, and the caller drops it.
 */
vv_status_t vv_cuda_visible_map(const char* text, const int* nodes,
                                int n_nodes, char* out, size_t out_size) {
    if (!out || out_size == 0) return VV_ERR_NULL_PTR;
    out[0] = '\0';
    if (!text || !nodes || n_nodes <= 0) return VV_ERR_UNSUPPORTED;

    /* UUIDs and MIG ids already name a device rather than a position. */
    for (const char* p = text; *p; p++)
        if (!isdigit((unsigned char)*p) && *p != ',' && !isspace((unsigned char)*p))
            return VV_ERR_UNSUPPORTED;

    size_t used = 0;
    int listed = 0, mapped = 0;
    const char* p = text;
    while (*p) {
        while (*p == ',' || isspace((unsigned char)*p)) p++;
        if (!*p) break;
        char* end = NULL;
        const long v = strtol(p, &end, 10);
        if (end == p) return VV_ERR_UNSUPPORTED;
        p = end;
        listed++;

        int pos = -1;
        for (int i = 0; i < n_nodes; i++)
            if (nodes[i] == (int)v) { pos = i; break; }
        /*
         * CUDA stops at the first entry it cannot resolve, so a list that is
         * partly wrong is worth no more than one that is entirely wrong.
         */
        if (pos < 0) return VV_ERR_NOT_FOUND;

        const int n = snprintf(out + used, out_size - used, "%s%d",
                               mapped ? "," : "", pos);
        if (n < 0 || (size_t)n >= out_size - used) return VV_ERR_INVALID_ARG;
        used += (size_t)n;
        mapped++;
    }
    if (listed == 0) return VV_ERR_UNSUPPORTED;
    return VV_OK;
}

/**
 * @brief Apply vv_cuda_visible_map() to this process, once, before CUDA runs.
 *
 * CUDA reads `CUDA_VISIBLE_DEVICES` when its runtime initializes and never
 * again, so the correction has to happen before the first CUDA call. Every
 * command reaches the device layer through vv_gpu_set_resolve() below, which
 * is what calls this.
 *
 * Linux only: the nodes under `/dev` are what says which cards this process
 * actually holds, and it is containers that renumber them. The once-flag is
 * the only process-wide state here, for a variable that is process-wide too.
 */
#ifdef __linux__
static pthread_once_t g_visible_once = PTHREAD_ONCE_INIT;

static void fix_visible_devices(void) {
    const char* keep = getenv("VV_KEEP_CUDA_VISIBLE_DEVICES");
    if (keep && keep[0] && keep[0] != '0') return;
    const char* cvd = getenv("CUDA_VISIBLE_DEVICES");
    if (!cvd || !cvd[0]) return;

    int nodes[VV_MAX_GPUS];
    int n_nodes = 0;
    DIR* d = opendir("/dev");
    if (!d) return;
    for (struct dirent* e = readdir(d); e && n_nodes < VV_MAX_GPUS;
         e = readdir(d)) {
        if (strncmp(e->d_name, "nvidia", 6) != 0 ||
            !isdigit((unsigned char)e->d_name[6]))
            continue;                            /* nvidiactl, nvidia-uvm, … */
        char* end = NULL;
        const long v = strtol(e->d_name + 6, &end, 10);
        if (!end || *end != '\0') continue;
        nodes[n_nodes++] = (int)v;
    }
    closedir(d);
    for (int i = 1; i < n_nodes; i++) {          /* ascending; a tiny list */
        const int v = nodes[i];
        int j = i - 1;
        while (j >= 0 && nodes[j] > v) { nodes[j + 1] = nodes[j]; j--; }
        nodes[j + 1] = v;
    }

    char fixed[256];
    const vv_status_t s = vv_cuda_visible_map(cvd, nodes, n_nodes,
                                              fixed, sizeof(fixed));
    if (s == VV_ERR_NOT_FOUND) {
        VV_LOG_W("cuda: CUDA_VISIBLE_DEVICES='%s' names no device this "
                 "container holds; ignoring it and using the %d that %s "
                 "mounted (VV_KEEP_CUDA_VISIBLE_DEVICES=1 keeps it)",
                 cvd, n_nodes, n_nodes == 1 ? "is" : "are");
        unsetenv("CUDA_VISIBLE_DEVICES");
    } else if (s == VV_OK && strcmp(fixed, cvd) != 0) {
        VV_LOG_W("cuda: CUDA_VISIBLE_DEVICES='%s' is in the host's numbering; "
                 "this container holds those cards as '%s', which is what is "
                 "used (VV_KEEP_CUDA_VISIBLE_DEVICES=1 keeps it)",
                 cvd, fixed);
        setenv("CUDA_VISIBLE_DEVICES", fixed, 1);
    }
}
#endif

void vv_cuda_fix_visible_devices(void) {
#ifdef __linux__
    pthread_once(&g_visible_once, fix_visible_devices);
#endif
}

/**
 * @brief The one sequence every command runs on its device flags.
 *
 * `--cpu` is looked at first, because it means "no device": evaluating
 * `--gpus all` anyway made a GPU-less container fail on a device list it
 * had just been told not to use -- which is what a scheduler hands a CPU-only
 * replica whose run command carries `--gpus all`. So with it nothing is
 * parsed or queried, and the log says the flags went unused.
 */
vv_status_t vv_gpu_set_from_flags(const char* gpus, const char* gpu_memory,
                                  bool cpu_only, int* gpu_id,
                                  vv_gpu_set_t* set) {
    if (!gpu_id || !set) return VV_ERR_NULL_PTR;
    if (cpu_only) {
        if (gpus || gpu_memory)
            VV_LOG_I("gpus: --cpu given; --gpus and --gpu-memory are ignored");
        return vv_gpu_set_resolve(set, *gpu_id, true);
    }

    vv_status_t s;
    if (gpus && (s = vv_gpu_set_parse(gpus, set)) != VV_OK) return s;
    if (gpu_memory && (s = vv_gpu_set_caps(gpu_memory, set)) != VV_OK)
        return s;
    if (set->n > 0) *gpu_id = set->id[0];
    return vv_gpu_set_resolve(set, *gpu_id, false);
}

/**
 * @brief What the device already holds that the budget cannot place.
 *
 * With an explicit cap this is the CUDA context, the driver's own
 * allocations and anything another process is holding, all of which count
 * against "this card will not go above 18 GiB". With no cap the budget is
 * already a fraction of what is *free*, so none of it is double-counted and
 * the answer is zero.
 */
size_t vv_gpu_reserved(const vv_gpu_set_t* set, int index) {
    if (!set || index < 0 || index >= set->n) return 0;
    size_t total = 0, freem = 0;
    if (vv_dev_get_device_info(set->id[index], &total, &freem, NULL) != VV_OK)
        return 0;
    if (vv_mem_cap_bytes(&set->cap[index], total) == 0) return 0;
    return total > freem ? total - freem : 0;
}

size_t vv_gpu_budget(const vv_gpu_set_t* set, int index, float vram_budget) {
    if (!set || index < 0 || index >= set->n) return 0;
    size_t total = 0, freem = 0;
    if (vv_dev_get_device_info(set->id[index], &total, &freem, NULL) != VV_OK)
        return 0;

    /* What is free, scaled by the legacy fraction... */
    size_t budget = (size_t)((double)freem * (double)vram_budget);
    /* ...and then never more than the operator allowed. */
    const size_t cap = vv_mem_cap_bytes(&set->cap[index], total);
    if (cap && cap < budget) budget = cap;
    return budget;
}
