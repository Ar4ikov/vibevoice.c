/**
 * @file wav.c
 * @brief WAV file parser supporting PCM 8/16/24/32-bit and float32.
 */

#include "vibevoice/audio.h"
#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

/* WAV format constants */
#define WAV_FORMAT_PCM   1
#define WAV_FORMAT_FLOAT 3

#pragma pack(push, 1)
typedef struct {
    char     riff_id[4];       /* "RIFF" */
    uint32_t file_size;
    char     wave_id[4];       /* "WAVE" */
} wav_header_t;

typedef struct {
    char     chunk_id[4];
    uint32_t chunk_size;
} wav_chunk_header_t;

typedef struct {
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
} wav_fmt_t;
#pragma pack(pop)

vv_status_t vv_audio_load_wav(const char* path, float** samples,
                               int* num_samples, int* sample_rate) {
    if (!path || !samples || !num_samples || !sample_rate) {
        return VV_ERR_NULL_PTR;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        VV_LOG_E("wav: cannot open '%s'", path);
        return VV_ERR_IO;
    }

    /* Read RIFF header */
    wav_header_t header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return VV_ERR_AUDIO_FORMAT;
    }

    if (memcmp(header.riff_id, "RIFF", 4) != 0 ||
        memcmp(header.wave_id, "WAVE", 4) != 0) {
        VV_LOG_E("wav: not a valid WAV file");
        fclose(f);
        return VV_ERR_AUDIO_FORMAT;
    }

    /* Find fmt and data chunks */
    wav_fmt_t fmt;
    memset(&fmt, 0, sizeof(fmt));
    bool found_fmt = false;
    const uint8_t* data_buf = NULL;
    uint32_t data_size = 0;
    uint8_t* raw_data = NULL;

    while (!feof(f)) {
        wav_chunk_header_t chunk;
        if (fread(&chunk, sizeof(chunk), 1, f) != 1) break;

        if (memcmp(chunk.chunk_id, "fmt ", 4) == 0) {
            size_t to_read = chunk.chunk_size < sizeof(fmt) ?
                             chunk.chunk_size : sizeof(fmt);
            if (fread(&fmt, to_read, 1, f) != 1) {
                fclose(f);
                return VV_ERR_AUDIO_FORMAT;
            }
            /* Skip any extra fmt bytes */
            if (chunk.chunk_size > to_read) {
                fseek(f, (long)(chunk.chunk_size - to_read), SEEK_CUR);
            }
            found_fmt = true;
        }
        else if (memcmp(chunk.chunk_id, "data", 4) == 0) {
            data_size = chunk.chunk_size;
            raw_data = (uint8_t*)vv_alloc(data_size);
            if (!raw_data) {
                fclose(f);
                return VV_ERR_OUT_OF_MEMORY;
            }
            if (fread(raw_data, 1, data_size, f) != data_size) {
                vv_free(raw_data);
                fclose(f);
                return VV_ERR_IO;
            }
            data_buf = raw_data;
        }
        else {
            /* Skip unknown chunk */
            fseek(f, (long)chunk.chunk_size, SEEK_CUR);
        }
    }
    fclose(f);

    if (!found_fmt || !data_buf) {
        VV_LOG_E("wav: missing fmt or data chunk");
        if (raw_data) vv_free(raw_data);
        return VV_ERR_AUDIO_FORMAT;
    }

    /* Validate format */
    if (fmt.audio_format != WAV_FORMAT_PCM &&
        fmt.audio_format != WAV_FORMAT_FLOAT) {
        VV_LOG_E("wav: unsupported format %d (only PCM and float supported)",
                 fmt.audio_format);
        vv_free(raw_data);
        return VV_ERR_AUDIO_FORMAT;
    }

    int n_channels = fmt.num_channels;
    int bps = fmt.bits_per_sample;
    int bytes_per_sample = bps / 8;
    int total_samples = (int)(data_size / (bytes_per_sample * n_channels));

    /* Convert to float32 mono */
    float* out = (float*)vv_alloc((size_t)total_samples * sizeof(float));
    if (!out) {
        vv_free(raw_data);
        return VV_ERR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < total_samples; i++) {
        float val = 0.0f;

        for (int ch = 0; ch < n_channels; ch++) {
            int offset = (i * n_channels + ch) * bytes_per_sample;
            float sample = 0.0f;

            if (fmt.audio_format == WAV_FORMAT_FLOAT && bps == 32) {
                memcpy(&sample, data_buf + offset, 4);
            }
            else if (fmt.audio_format == WAV_FORMAT_PCM) {
                if (bps == 8) {
                    /* 8-bit PCM is unsigned */
                    sample = ((float)data_buf[offset] - 128.0f) / 128.0f;
                }
                else if (bps == 16) {
                    int16_t s16;
                    memcpy(&s16, data_buf + offset, 2);
                    sample = (float)s16 / 32768.0f;
                }
                else if (bps == 24) {
                    int32_t s32 = 0;
                    memcpy(&s32, data_buf + offset, 3);
                    /* Sign extend */
                    if (s32 & 0x800000) s32 |= 0xFF000000;
                    sample = (float)s32 / 8388608.0f;
                }
                else if (bps == 32) {
                    int32_t s32;
                    memcpy(&s32, data_buf + offset, 4);
                    sample = (float)s32 / 2147483648.0f;
                }
            }
            val += sample;
        }

        /* Downmix to mono by averaging channels */
        out[i] = val / (float)n_channels;
    }

    vv_free(raw_data);

    *samples = out;
    *num_samples = total_samples;
    *sample_rate = (int)fmt.sample_rate;

    VV_LOG_I("wav: loaded %d samples @ %d Hz, %d ch, %d bits",
             total_samples, (int)fmt.sample_rate, n_channels, bps);
    return VV_OK;
}
