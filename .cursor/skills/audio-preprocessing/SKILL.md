# Skill: Audio Preprocessing Pipeline

## Purpose
Transform raw audio into 24kHz normalized PCM for the Conv-VAE tokenizer encoders.
**NO mel spectrogram, NO FFT, NO STFT.** The model takes raw PCM directly.

## Pipeline
1. WAV parser: read PCM16/float32, mono/stereo → mono float32
2. Resample: any sample rate → 24kHz (sinc/polyphase interpolation)
3. Normalize: RMS normalize to target_dB_FS = -25, eps = 1e-6

## Key API
```c
vv_status_t vv_audio_load_wav(const char* path, float** out, int* n_samples, int* sr);
vv_status_t vv_audio_resample(const float* in, int in_sr, int in_len,
                               float** out, int* out_len, int target_sr);
vv_status_t vv_audio_normalize(float* samples, int n_samples,
                                float target_db_fs, float eps);
```

## Config (from preprocessor_config.json)
- target_sample_rate: 24000
- normalize_audio: true
- target_dB_FS: -25
- eps: 1e-6
- speech_tok_compress_ratio: 3200 (informational)
