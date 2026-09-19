# CLAUDE.md — vibevoice.c

> Контекстный файл для AI-ассистента. Содержит всё необходимое для понимания
> проекта, принятия архитектурных решений и генерации кода.
> Данные верифицированы по исходникам: modeling_vibevoice_asr.py,
> configuration_vibevoice.py, config.json, preprocessor_config.json.

---

## 0. Статус (проверено на RTX 3090 + Ryzen 9 5900X, CUDA 12.4)

Runtime работает end-to-end и **посимвольно совпадает с PyTorch-эталоном**
(`transformers` + `bitsandbytes`, тот же чекпоинт) на 11 c, 120 c (два
сегмента, два говорящих) и 32 минутах.

| Метрика | Факт |
|---|---|
| Загрузка модели | **4.0 c** (NF4), 5.5 c (AWQ, включая репак; ещё 0.3–1.2 c на GPU-раскладку W4A16); потоки = физические ядра |
| Speech encoding | **50 мс** на 11 c аудио, 352 мс на 120 c, 5.4 c на 32 мин |
| Prefill | **3056 tok/s** (14449-токенный промпт, tensor cores) |
| Decode | **131 tok/s** (0.2K ctx), 123 (1.5K), 93 (14K→24K); flashinfer 132 / 129 / 107 |
| RTF | **0.046** (120 c), **0.058** (32 мин), 0.051 с `--attn flashinfer` |
| VRAM | 9.8 GB (3.2 веса + 1.8 KV на 32K) |
| CPU-only | prefill 77 tok/s (991 GFLOP/s), decode 7.9 tok/s, RTF 0.81 |

Для сравнения: `transformers` + `bitsandbytes` на той же карте — 27.6 tok/s.

Релизный бинарник: **10.7 MB**, зависит только от libc и libm. Статический
cudart, cuBLAS не используется (свои WMMA-ядра, быстрее cuBLAS на этих
формах), cubin для sm_75..sm_90 + PTX. Стартует и без драйвера NVIDIA —
тогда работает CPU-путь.

### Что умеет

* `vv_cli` — транскрипция файла; `serve` — OpenAI-совместимый HTTP
  (совместим с бэкендом speech-to-text в GPUStack); `chat` — интерактивный
  цикл с горячей моделью; `mic` — живая транскрипция с VAD.
* Веса: NF4 (bitsandbytes), AWQ, GPTQ. AWQ/GPTQ перепаковываются при
  загрузке в row-major INT4G (иначе GEMV не коалесится) с точной целой
  нулевой точкой; GPU-путь затем переставляет нибблы внутри 16-байтных
  отрезков строки (`vv_model_int4g_to_gpu_layout`, одна копия) под
  `src/cuda/w4a16.cu`: GEMV с LOP3/HFMA2 (q/k/v и gate/up — одним запуском)
  и Marlin-подобный GEMM на tensor cores для M > 1, без разворачивания
  веса в FP16. Деквант — `(q − z)·s`, как в эталоне. GPTQ act-order
  (`desc_act`): каналы входа сортируются по группам при загрузке
  (`vv_weight_t.perm`), активации собираются в том же порядке
  (`vv_w4a16_gather_dev`). `VV_INT4G_LEGACY=1` — старый путь.
* Неквантованные чекпоинты (BF16/F16/F32, `microsoft/VibeVoice-ASR`):
  плотный FP16 или `--quant nf4|int4|int8` прямо при загрузке из mmap, до
  placement. Формат проекции определяется по dtype в файле, а не по имени
  (U8+`.absmax` → NF4, I32 `qweight` → INT4G, float → dense); каждый тензор
  сверяется с формой из config, отсутствующий/кривой — ошибка с именем.
  `tie_word_embeddings` — head и embedding один буфер. BF16 dense на 3090:
  56 tok/s, 18.7 GB; `--quant int4` — 149 tok/s (W4A16-ядра), `int8` (per-channel) —
  97 tok/s, 12.5 GB; те же слова, что NF4.
* Семейства моделей (`include/vibevoice/family.h`): `asr-7b`, `asr-bitnet`,
  `asr-streaming-7b` — промпт, стоп-токены, нормализация, геометрия чанков.
  Один проход по слоям — `vv_layer_tensors()`; prefill дописывает с
  `kv->current_len` (GPU и CPU), CPU prefill идёт чанками.
* **VibeVoice-ASR-Streaming-7B** (`docs/STREAMING.md`, `include/vibevoice/stream.h`,
  `src/inference/stream_ctx.c`): сессия на контексте — промпт в сброшенный
  KV, каждое окно 26 кадров (22 + 4 lookahead) — stateless-задача фронтенда,
  коннекторы пишут прямо в строки чанка, prefill `[<|text_chunk_end|>]
  <|object_ref_start|> 26 фич <|object_ref_end|>` с `kv->current_len`,
  greedy decode на захваченном графе (графы живут в сессии и переживают
  prefill'ы). Несколько готовых окон кодируются одним заходом
  (`encode_ahead`). `vv_inference_transcribe` для этой модели идёт через
  сессию; `vv_cli --audio` печатает чанки по мере готовности, `mic` —
  живой поток без VAD, `serve` — SSE `stream=true` и WebSocket
  `/v1/audio/stream` (слот на сессию, отключение клиента отменяет сессию).
  `--quant none` посимвольно совпадает с upstream `streaming_generate` на
  jfk/test30/test120 (156/156 чанков); int4 на 3090 — 92 мс на чанк,
  RTF 0.032 (32 мин — 114 мс, 0.042); `serve` держит ~24 живых
  int4-потока на 3090 (p95 < 300 мс) — предел time-slicing: каждый слот
  декодирует сам, батча между слотами нет. Живая сессия резервирует KV на
  `--stream-reserve` секунд при открытии (иначе 503 / close 1013), idle
  считается по аудио, а не по пингам, слот ждётся не дольше
  `--stream-slot-wait`.
  Нормализация громкости для этой модели выключена.
* KV-кэш: `--kv-cache fp16|fp8|fp8-e5m2|tq4|tq3|tq2|tq1.5`. На 32K позиций
  1792 → 896 / 462 / 350 / 238 / 182 MB. fp8, fp8-e5m2 и tq4 дают
  идентичный транскрипт.
* `--gpu-layers N` — сколько слоёв держать в VRAM, остальные стримятся с
  перекрытием копирования и pinned-памятью. Полный стриминг на PCIe 4.0 x16
  даёт 6.6 tok/s при 21.8 GB/s — это насыщенная шина.
* `--slots N` в сервере — N одновременных запросов на одной копии весов.
  Слоты одного устройства делят пул KV-страниц по 64 позиции
  (`--kv-paged auto|on|off`), а не держат по окну каждый.
* `--attn auto|fa1|fa2|flashinfer` (`VV_ATTN=`): бэкенды внимания за одной
  точкой диспетчеризации (`vv_attn_prefill/decode`). `auto` = fa2 —
  побитово совпадает со старыми ядрами (тесты сравнивают декод fa2 и fa1
  бит в бит), поэтому транскрипты не двигаются (у Streaming-7B `auto` =
  flashinfer: split-KV для 29-строчного prefill на растущем кэше, JSON тот
  же, на 32 мин быстрее на 15%). flashinfer быстрее
  (tensor cores и в декоде, fp8/tq через те же ядра), но округляет P и
  суммирует в своём порядке: слова те же, таймстемпы сдвигаются на 10–20 мс.
* `--gpus 0,1|all` выбирает устройства, `--gpu-memory 80%|18GiB|8192M|байты`
  (одно значение на все или по одному на устройство) ограничивает, сколько
  на каждом можно занять. Лимит учитывает контекст CUDA, веса speech-энкодера
  и его scratch, и влияет на выбор placement, а не проверяется постфактум.
  `serve` кладёт по реплике на каждое устройство и распределяет слоты: восемь
  30-секундных файлов через 4 слота — 13.2 с на одной 3090 и 7.6 с на двух.
  `--split-mode auto|replica|layer`: реплика на каждое устройство или одна
  модель, разрезанная по слоям (слои 0..k на первом, остальные дальше, KV
  своих слоёв — на своём устройстве, между шардами ходит только hidden
  state). `auto` = реплики, пока на каждое устройство есть слот, иначе
  шардирование. Две 3090 по 6 GiB: 30-секундный файл идёт от 7/28 слоёв
  резидентно и RTF 0.595 до 28/28 и RTF 0.059.

### Что было сломано (и почему это стоит помнить)

Четыре независимых дефекта, каждый из которых в одиночку превращал вывод в мусор:

1. **BPE-токенизатор** не делал byte-level кодирование (нет GPT-2 алфавита,
   нет pre-tokenizer split), плюс O(merges × tokens) перебор слияний.
2. **Causal SConv1d**: левый паддинг брался как `k - 1` вместо
   `(k - 1) * dilation - (stride - 1)`. Для stride=1 совпадает, поэтому баг
   был не виден на stem/head, но сдвигал каждый downsample-слой.
3. **FFN в tokenizer** применял SiLU там, где эталон использует точный GELU
   (`erf`, не tanh-аппроксимация).
4. **Flash-attention prefill** вычислял границу цикла по KV из построчного
   causal-лимита — варпы одного блока приходили к разному числу
   `__syncthreads()` и разносили общие K/V-тайлы.

Позже нашлись ещё три:

5. **Квантование K «как есть»** ломает вывод на любой разрядности, включая
   FP8 (cos attention 0.86). У Qwen2 ключи имеют огромную общую компоненту:
   ‖k‖ = 273 против ‖k − mean‖ = 11. Softmax инвариантен к сдвигу всех
   ключей, поэтому кэш хранит K относительно опорного вектора слоя.
6. **`vv_layer_prefetch_wait` определял резидентность слоя по `tensor.on_gpu`**,
   который выставляет и staging — из-за этого compute-поток не ждал только
   что начатое копирование. Проявилось после перехода на pinned-память.
7. **Staging-буфер слоя** резервировал выравнивание на 16 тензоров при 20
   живых — запись за границу слота.
8. **`device.h` в `conv_vae.c` включался только в ветке `#else` (не-Windows).**
   В MSVC-сборке весь GPU-энкодер вызывался без прототипов: C подразумевает
   `int f()` и продвигает `float` до `double`, а на Windows x64 аргументы
   позиционные — так терялись `eps` в RMSNorm и `alpha`/`beta` в GEMM (первый
   FFN-GEMM писал нули). Энкодер «успешно» отрабатывал, ошибок не было, а
   модель транскрибировала любую речь как `[Noise]`/`[Music]`. На Linux
   (SysV ABI) целочисленные и SSE аргументы нумеруются независимо, поэтому
   там же собранный код работал верно. Теперь неявное объявление функции —
   ошибка сборки (`/we4013`, `-Werror=implicit-function-declaration`), а
   `VV_ENC_STATS=1` печатает контрольные суммы энкодера по стадиям, чем эта
   разница и была найдена.

### Формат промпта (точно как в `vibevoice_asr_processor.py`)

```
<|im_start|>system

You are a helpful assistant that transcribes audio input into text output in JSON format.<|im_end|>

<|im_start|>user

<|object_ref_start|>[<|box_start|> × ceil(N/3200)]<|object_ref_end|>

This is a {dur:.2f} seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>

```

**Generation prompt НЕ добавляется** — модель сама генерирует
`<|im_start|>assistant
`, а затем JSON-массив сегментов. С hotwords
инструкция принимает вид `...seconds audio, with extra info: {ctx}

Please
transcribe it with these keys: ...`.

Спец-токены (переиспользованные из Qwen2.5):
`speech_start = <|object_ref_start|>` (151646),
`speech_end = <|object_ref_end|>` (151647),
`speech_pad = <|box_start|>` (151648).

### Что ещё не сделано

* **Metal.** На macOS работает CPU-путь (NEON, проверен под qemu-aarch64).
  Бэкенд Metal не написан: его нельзя ни собрать, ни запустить с машины
  разработки. Шов готов — `include/vibevoice/device.h` объявляет набор
  операций, сборка линкует ровно одну реализацию, `src/device/device_none.c`
  показывает форму.
* Акустический латент по умолчанию берётся как среднее (детерминированно).
  Выборка эталона есть: `--acoustic-sampling gaussian|fix --seed N`.

---

## 1. Что это за проект

**vibevoice.c** — высокопроизводительный runtime на **чистом C** для запуска
квантованной (4-bit NF4) модели **VibeVoice-ASR** от Microsoft.

Модель выполняет:
- Автоматическое распознавание речи (ASR) длительностью до 60 минут
- Диаризацию (определение, кто говорит)
- Генерацию временных меток
- Поддержку пользовательских hotwords (ключевых слов)

**Ключевое ограничение**: весь runtime — на C. Никакого Python, PyTorch,
TensorFlow, ONNX Runtime в runtime-коде. Python допустим только в `tools/`
для одноразовой конвертации весов.

---

## 2. Ссылки на модель и ресурсы

| Ресурс | URL |
|--------|-----|
| Оригинальное репо | https://github.com/microsoft/VibeVoice |
| Оригинальные веса (BF16) | https://huggingface.co/microsoft/VibeVoice-ASR |
| 4-bit квантованные веса | https://huggingface.co/scerz/VibeVoice-ASR-4bit |
| AWQ W4A16 (asym) веса | https://huggingface.co/Ar4ikov/VibeVoice-ASR-AWQ-W4A16-ASYM |
| Технический отчёт ASR | https://arxiv.org/pdf/2601.18184 |
| Лицензия модели | MIT |

---

## 3. Архитектура модели VibeVoice-ASR (ВЕРИФИЦИРОВАНО)

### Общая схема
```
Raw Audio (any sample rate)
    │
    ▼
┌──────────────────────────────────┐
│  Resample to 24kHz mono          │  target_sample_rate: 24000
│  Normalize to -25 dBFS           │  normalize_audio: true
└──────────────────────────────────┘
    │                    │
    ▼                    ▼
┌──────────────┐  ┌──────────────┐
│  Acoustic     │  │  Semantic     │
│  Tokenizer    │  │  Tokenizer    │
│  Encoder      │  │  Encoder      │
│  (Conv-VAE)   │  │  (Conv-VAE)   │
│  vae_dim=64   │  │  vae_dim=128  │
│  gaussian     │  │  deterministic│
│  sampling     │  │  (mean only)  │
│  FP16/BF16    │  │  FP16/BF16    │
└──────┬───────┘  └──────┬───────┘
       │                  │
       ▼                  ▼
┌──────────────┐  ┌──────────────┐
│  Acoustic     │  │  Semantic     │
│  Connector    │  │  Connector    │
│  MLP          │  │  MLP          │
│  64 → 3584    │  │  128 → 3584   │
│  FP16         │  │  FP16         │
└──────┬───────┘  └──────┬───────┘
       │                  │
       └───────┬──────────┘
               ▼
       Element-wise Add
               │
               ▼
┌──────────────────────────────────┐
│  LLM Backbone (Qwen2-7B)        │
│  28 layers, decoder-only         │
│  NF4 quantized (4-bit)           │
│  GQA: 28 Q heads, 4 KV heads    │
│  hidden=3584, head_dim=128       │
│  max_position_embeddings=131072  │
└──────────────────────────────────┘
               │
               ▼
       Structured Transcription (JSON)
```

### ВАЖНО: НЕТ mel-спектрограммы!
Модель НЕ использует mel/STFT/FFT. Сырой 24kHz PCM подаётся напрямую
в два Conv-VAE токенизатора. Каждый сжимает аудио в 3200 раз
(произведение ratios [8,5,5,4,2,2] = 3200), что даёт 24000/3200 = 7.5 Hz.

### Conv-VAE Tokenizer Encoder (из config.json)
```
Acoustic Tokenizer:
  channels: 1 (mono input)
  causal: true
  encoder_ratios: [8, 5, 5, 4, 2, 2]     (6 downsample stages)
  encoder_depths: "3-3-3-3-3-3-8"          (7 stage groups)
  encoder_n_filters: 32                     (base filter count)
  vae_dim: 64                               (latent dim)
  fix_std: 0.5                              (gaussian sampling std)
  std_dist_type: "gaussian"
  mixer_layer: "depthwise_conv"
  layernorm: "RMSNorm" (eps=1e-5)
  layer_scale_init_value: 1e-6

Semantic Tokenizer:
  (same architecture but)
  vae_dim: 128
  fix_std: 0                                (deterministic, no sampling)
  std_dist_type: "none"
```

### Speech Connector (из modeling_vibevoice.py SpeechConnector)
```python
# fc1: Linear(vae_dim, hidden_size)
# norm: LlamaRMSNorm(hidden_size, eps=1e-6)
# fc2: Linear(hidden_size, hidden_size)
# forward: x -> fc1 -> RMSNorm -> fc2  (NO activation, NO GELU!)
```

### Параметры LLM (Qwen2-7B backbone, из config.json)
```
model_type:              "qwen2"
hidden_size:             3584
num_hidden_layers:       28
num_attention_heads:     28       (query heads)
num_key_value_heads:     4        (GQA: grouped-query attention)
head_dim:                128      (= 3584 / 28)
intermediate_size:       18944    (MLP intermediate)
vocab_size:              152064
max_position_embeddings: 131072   (128K context!)
rope_theta:              1000000.0
rms_norm_eps:            1e-6
hidden_act:              "silu"   (SwiGLU in MLP)
layer_types:             all "full_attention" (no sliding window)
use_mrope:               false
```

### Квантизация (4-bit NF4 — bitsandbytes)
- **Что квантовано**: ТОЛЬКО Linear-слои в Qwen2 LLM
  (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- **Что НЕ квантовано**: acoustic/semantic tokenizer encoders,
  connectors, embed_tokens, layernorm weights, lm_head — всё в BF16/FP16.
- **Формат**: NormalFloat4 (NF4) — 16 уровней.
- **Упаковка**: 2 значения в 1 байт (uint8).
- **Block size**: 64 элемента.
- **Double quantization**: scale каждого блока квантуется в FP8 (E4M3).
- **Dequantization**: `value_fp16 = NF4_LOOKUP[nibble] * block_scale`
- **Размер 4-bit модели**: ~7.66 GB total (2 safetensors файла).

### Audio Preprocessing (из preprocessor_config.json)
```
processor_class:         "VibeVoiceASRProcessor"
target_sample_rate:      24000         (24kHz, НЕ 16kHz!)
speech_tok_compress_ratio: 3200        (24000 / 7.5 Hz)
normalize_audio:         true
target_dB_FS:            -25
eps:                     1e-6
```

### Специальные токены
```
<|startoftranscript|>    — начало транскрипта
<|endoftranscript|>      — конец (stop token для decode)
<|speaker_1|> ... <|speaker_N|>  — идентификация спикера
<|timestamp_X.XX|>       — временная метка
<|hotwords|>             — начало секции hotwords
<|nospeech|>             — тишина/без речи
```

---

## 4. Целевая платформа

| Параметр | Значение |
|----------|----------|
| ОС | Windows 11 x64 (основная), Linux x64 (будущее) |
| GPU | NVIDIA Ampere (SM 8.0+): RTX 3060 12GB, 3070, 3080, 3090 |
| Min VRAM | 12 GB (для 4-bit, с KV-cache на 30 минут) |
| Компилятор | MSVC 2022 (v143), nvcc (CUDA 12.2+) |
| Сборка | CMake 3.28+ |
| CUDA | 12.2+ (Tensor Cores FP16/INT8 на Ampere) |
| cuBLAS | Через CUDA Toolkit |
| cuDNN | 9.x (опционально, для Conv-VAE оптимизации) |

---

## 5. Структура проекта

```
vibevoice.c/
│
├── CMakeLists.txt                  # Корневой CMake
├── CLAUDE.md                       # ← Этот файл
├── .cursorrules                    # Правила для Cursor IDE
├── .cursor/skills/                 # Agent Skills (навыки AI)
│
├── include/vibevoice/              # Публичные C заголовки
│   ├── vibevoice.h                 # Главный заголовок (umbrella)
│   ├── types.h                     # Базовые типы, коды ошибок
│   ├── audio.h                     # Audio preprocessing API
│   ├── text_tokenizer.h            # BPE text tokenizer API
│   ├── tokenizer_encoder.h         # Conv-VAE speech tokenizer encoder API
│   ├── connector.h                 # Speech connector MLP API
│   ├── model.h                     # Model loading API
│   ├── inference.h                 # Inference pipeline API
│   ├── frontend.h                  # Speech front end устройства: энкодеры + коннекторы + батчинг
│   └── profiler.h                  # Profiling utilities API
│
├── src/
│   ├── core/                       # Ядро: аллокаторы, логгер, ошибки
│   │   ├── alloc.c                 # Обёртки аллокации (CPU/GPU)
│   │   ├── logger.c                # Логгирование (уровни: ERROR/WARN/INFO/DEBUG)
│   │   └── error.c                 # Коды ошибок, строковые описания
│   │
│   ├── audio/                      # Аудио обработка (БЕЗ mel/FFT!)
│   │   ├── wav.c                   # WAV парсер (PCM 16-bit, float32)
│   │   ├── resample.c              # Ресемплер (sinc/polyphase → 24kHz)
│   │   └── normalize.c             # RMS нормализация к -25 dBFS
│   │
│   ├── text_tokenizer/             # BPE текстовый токенизатор
│   │   ├── bpe.c                   # BPE encode/decode
│   │   ├── vocab.c                 # Vocab + merges загрузка
│   │   └── special_tokens.c        # Специальные токены VibeVoice
│   │
│   ├── tokenizer_encoder/          # Conv-VAE speech tokenizer encoders
│   │   ├── conv_vae.c              # описание архитектуры, init, число кадров
│   │   ├── vae_plan.h              # план слоёв: контекст, шаг, хвост стриминга
│   │   ├── vae_gpu.c               # веса/состояние/арена, батчевый encode
│   │   └── vae_cpu.c               # CPU-энкодер: тайлы по времени, OpenMP, packed GEMM
│   │
│   ├── connector/                  # Speech connectors (MLP)
│   │   └── speech_connector.c      # fc1 → RMSNorm → fc2 (no activation), CPU и GPU
│   │
│   ├── model/                      # Загрузка модели
│   │   ├── safetensors.c           # Парсер формата safetensors
│   │   ├── config.c                # Парсер config.json
│   │   └── loader.c                # Оркестрация загрузки всех компонентов
│   │
│   ├── quant/                      # Квантизация
│   │   ├── nf4_table.c             # NF4 lookup table (16 значений)
│   │   └── dequant_cpu.c           # CPU reference dequantization
│   │
│   ├── cuda/                       # CUDA ядра (.cu файлы)
│   │   ├── dequant_nf4.cu          # NF4 dequantization kernel
│   │   ├── vae_kernels.cu          # Conv-VAE: свёртки по таблице дескрипторов, fused mixer, GEMM с эпилогами
│   │   ├── rmsnorm.cu              # RMSNorm kernel
│   │   ├── rope.cu                 # Rotary Position Embeddings
│   │   ├── attention.cu            # GQA Flash Attention (28Q/4KV)
│   │   ├── swiglu.cu               # SwiGLU activation
│   │   ├── gemm.cu                 # GEMM wrappers (cuBLAS + NF4)
│   │   ├── embedding.cu            # Embedding lookup
│   │   └── cuda_utils.cu           # Memory management, stream helpers
│   │
│   ├── device/                     # device_none.c — заглушка без ускорителя
│   ├── engine/                     # пул слотов над одной копией весов
│   ├── server/                     # HTTP + OpenAI-совместимый API
│   └── inference/                  # Inference pipeline
│       ├── pipeline.c              # Полный pipeline: audio → transcript
│       ├── frontend.c              # Speech front end на устройство, сервис батчинга
│       ├── decoder.c               # Autoregressive decoder loop
│       ├── kv_cache.c              # KV-cache management (paged)
│       ├── sampling.c              # Token sampling (greedy, top-k)
│       └── postprocess.c           # Token stream → JSON transcription
│
├── cli/
│   ├── main.c                      # транскрипция файла + диспетчер команд
│   ├── cmd_serve.c                 # vv_cli serve
│   └── cmd_chat.c                  # vv_cli chat / vv_cli mic
│
├── tools/                          # Python утилиты (НЕ runtime)
│   ├── convert_weights.py          # HF safetensors → .vvmodel
│   ├── validate_weights.py         # Сравнение C vs Python output
│   └── requirements.txt            # Python зависимости
│
├── tests/                          # Тесты (C)
│   ├── test_audio.c                # resample + normalize
│   ├── test_safetensors.c          # parser correctness
│   ├── test_nf4.c                  # dequant vs CPU reference
│   ├── test_conv_vae.c             # tokenizer encoder vs Python
│   ├── test_vae_stream.c           # батч, стриминг, окна, GPU vs CPU (крошечная модель)
│   └── test_e2e.c                  # end-to-end: audio → transcript
│
├── bench/                          # Бенчмарки
│   └── benchmark.c
│
├── third_party/                    # Минимальные зависимости
│   └── cjson/                      # cJSON (MIT) — JSON парсер
│
└── LICENSES/                       # Лицензии зависимостей
    ├── MIT_VibeVoice.txt
    └── MIT_cJSON.txt
```

---

## 6. Правила разработки

### Код
- **Язык**: C11 (`/std:c11` для MSVC). CUDA C в .cu файлах.
- **Naming**: `snake_case`, префикс `vv_` для всех публичных символов.
- **Ошибки**: все функции возвращают `vv_status_t`. Никаких exceptions.
- **Память**: явный lifetime. `vv_alloc()` / `vv_free()`. Zero malloc in hot path.
- **GPU**: память выделяется при `init()`, переиспользуется. Pinned memory для transfers.
- **Streams**: минимум 2 CUDA stream (compute + transfer), overlap.
- **Комментарии**: Doxygen `/** */` для публичного API. Английский язык.

### Запрещено
- `malloc()` / `free()` напрямую (только через `vv_alloc` / `vv_free`).
- Глобальные переменные (кроме thread-local логгера и thread-local кэша
  неизменяемых свойств устройства — compute capability, число SM — по
  номеру устройства, как `att_device()` в `attention_fi.cu` и выбор ядра в
  `attention.cu`).
- `#include <python.h>` или любые Python/ML-framework headers.
- Хардкодить пути. Все пути — через параметры или env vars.
- `cudaMalloc` в hot path (только при init / resize).
- `printf` для ошибок (только `vv_log()`).

### CUDA ядра
- Все public функции: `extern "C"`.
- Все kernel-launch функции принимают `cudaStream_t`.
- Target architectures: sm_80 (Ampere), sm_86, sm_89 (Ada).
- `--use_fast_math` для Release builds.
- NVTX маркеры в каждом kernel-launch wrapper.

### Speech-энкодер
- Веса FP16 на устройстве — одна копия на устройство (`vv_vae_weights_t`),
  общая для всех слотов; состояние стрима (`vv_vae_state_t`) — хвосты свёрток;
  scratch — арена, размер которой считается из (max items, max samples) и
  входит в бюджет placement.
- Внутри encode — только запуски ядер на переданный stream: ни аллокаций,
  ни `cudaFree`, ни синхронизаций (кроме `VV_ENC_STATS=1`).
- Батч, стриминг чанками любой длины и stateless-окна дают тот же результат
  бит в бит, что и одиночный проход; `tests/test_vae_stream.c` это проверяет.
- Порядок суммирования не меняется: downsample и head — FP32 GEMM по im2col
  в порядке (канал, тап) прямой свёртки, форма тайла выбирает только поток,
  а не порядок. Split-K и tensor-core для свёрток сдвигали таймстемпы на
  32-минутном файле (4 из 168 сегментов на 10 мс) — поэтому их нет.
- Сервис фронтенда планирует по одному запуску, а не по задаче; его стримы
  создаются с фоновым приоритетом (`vv_dev_stream_create_background`), чтобы
  decode других слотов не ждал энкодер длинного файла.

---

## 7. Стратегия ускорения

### Приоритеты (от высшего к низшему)
1. **Батчевый и fused CUDA Conv-VAE** (упакованная ось времени, эпилоги GEMM, общая арена).
2. **Custom CUDA kernels** для LLM decoder (4-bit dequant + GEMM).
3. **cuBLAS** для FP16 GEMM (после dequant, и для connectors).
4. **Flash Attention** (custom kernel) для long-context (128K).
5. **CUDA Graphs** для стационарного decode loop.
6. **FP8 KV-cache** для экономии памяти.
7. **Kernel fusion** (RMSNorm + QKV projection, dequant + GEMM).

### Почему не TensorRT
Заглушки `src/trt/` удалены: энкодер упирался не в арифметику, а в
запуски, аллокации и синхронизации, которые TensorRT не убирает, а
`libnvinfer` ломал обещание «только libc и libm». `--trt-acoustic` и
`--trt-semantic` ещё один релиз разбираются и игнорируются с предупреждением.

### Memory Budget (RTX 3080, 10 GB usable)
```
Model weights (NF4 packed + FP16 non-quant): ~7.7 GB loaded, ~5.5 GB on GPU
Tokenizer encoder weights (FP16):            ~0.3 GB
Connector weights (FP16):                    ~0.1 GB
KV-cache (FP16, 30 min audio):              ~2.5 GB
Activations / workspace:                     ~1.0 GB
Audio buffers + misc:                        ~0.3 GB
──────────────────────────────────────────
Total:                                       ~9.7 GB
```

Для RTX 3060 12GB — с запасом 2.3 GB.
Для 60-мин аудио на 12 GB: FP8 KV-cache (~1.3 GB вместо 2.5 GB).

---

## 8. Pipeline инференса (подробно)

### Speech Encoding Phase
```
1. Load audio:              WAV → float32 PCM
2. Resample:                any SR → 24kHz mono
3. Normalize:               RMS normalize to -25 dBFS
4. Acoustic encoding:       raw PCM → acoustic_tokenizer.encode()
                            → [B, T, 64] mean + gaussian sample (std=0.5)
5. Semantic encoding:       raw PCM → semantic_tokenizer.encode()
                            → [B, T, 128] mean (deterministic)
6. Acoustic connector:      [B, T, 64]  → fc1 → RMSNorm → fc2 → [B, T, 3584]
7. Semantic connector:      [B, T, 128] → fc1 → RMSNorm → fc2 → [B, T, 3584]
8. Combine:                 acoustic_features + semantic_features → [B, T, 3584]
```

### Prefill Phase
```
1. Build input sequence:    [special_tokens, audio_features, prompt_tokens]
2. Embed text tokens:       embed_tokens(input_ids) → [B, S, 3584]
3. Replace audio positions: inputs_embeds[acoustic_input_mask] = combined_features
4. LLM prefill:             process full sequence through 28 Qwen2 layers
   - For each layer:
     - RMSNorm → Q, K, V (4-bit dequant + GEMM)
     - RoPE → GQA Attention → O proj → Residual
     - RMSNorm → SwiGLU MLP (4-bit dequant + GEMM) → Residual
   - Fill KV-cache for all positions
5. Get first output logits
```

### Decode Phase (autoregressive)
```
for each token:
  1. Embedding lookup: token_id → hidden [1, 1, 3584]
  2. For each layer (28x):
     a. RMSNorm(hidden)
     b. Q = dequant_gemm(hidden, W_q)     — NF4
        K = dequant_gemm(hidden, W_k)     — NF4
        V = dequant_gemm(hidden, W_v)     — NF4
     c. RoPE(Q, K)
     d. KV-cache: append K, V at position
     e. Attention(Q, K_cached, V_cached)   — GQA (28Q/4KV)
     f. O = dequant_gemm(attn_out, W_o)   — NF4
     g. hidden += O  (residual)
     h. RMSNorm(hidden)
     i. gate = dequant_gemm(hidden, W_gate) — NF4
        up   = dequant_gemm(hidden, W_up)   — NF4
     j. down = dequant_gemm(SwiGLU(gate, up), W_down) — NF4
     k. hidden += down (residual)
  3. Final RMSNorm
  4. LM Head: hidden → logits [152064]    — FP16 GEMM
  5. Greedy/top-k sampling → next token_id
  6. If token_id == <|endoftranscript|>: break
```

### Per-layer GEMM operations (NF4)
```
Layer projections (per transformer layer):
  W_q:    [3584, 3584]
  W_k:    [3584, 512]     (4 KV heads × 128)
  W_v:    [3584, 512]
  W_o:    [3584, 3584]
  W_gate: [3584, 18944]
  W_up:   [3584, 18944]
  W_down: [18944, 3584]

Total NF4 GEMMs per token: 28 layers × 7 = 196 GEMM operations
```

---

## 9. Model Files (4-bit, from HuggingFace)

### Files in scerz/VibeVoice-ASR-4bit
```
config.json                           (4.32 kB)
generation_config.json                (73 B)
preprocessor_config.json              (189 B)
model.safetensors.index.json          (232 kB)
model-00001-of-00002.safetensors      (4.97 GB)
model-00002-of-00002.safetensors      (2.69 GB)
```

### Weight Categories (from model.safetensors.index.json)
```
model.acoustic_tokenizer.encoder.*     — BF16, Conv-VAE encoder
model.acoustic_tokenizer.decoder.*     — BF16, Conv-VAE decoder (NOT needed for ASR)
model.semantic_tokenizer.*             — BF16 (check if present)
model.acoustic_connector.*             — BF16, MLP (fc1, fc2, norm)
model.semantic_connector.*             — BF16, MLP
model.embed_tokens.weight              — BF16, embedding table
model.layers.N.self_attn.{q,k,v,o}_proj.* — NF4 (uint8 packed + scales)
model.layers.N.mlp.{gate,up,down}_proj.*  — NF4
model.layers.N.input_layernorm.weight     — BF16
model.layers.N.post_attention_layernorm.weight — BF16
model.norm.weight                      — BF16
lm_head.weight                         — BF16 (may be tied to embed_tokens)
```

### NOTE: Tokenizer files not in 4-bit repo
Text tokenizer files (tokenizer.json, vocab, merges) must be fetched from
the base model microsoft/VibeVoice-ASR or Qwen2.5-7B.

---

## 10. Сборка

### Linux (основная площадка разработки — gpubox)

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j 16
VV_TEST_MODEL=./model_hf ctest --test-dir build --output-on-failure
```

`ctest` без `VV_TEST_MODEL` тоже проходит — тесты, которым нужны веса,
рапортуют SKIP. Всего 32 записи (attention-наборы идут по разу на бэкенд).

Streaming-7B: `VV_TEST_STREAM_MODEL=<каталог модели>` включает проверки
токенизатора, `VV_TEST_STREAM_REF=<dump-stream>[:<dump>...]` — сверку
текстов чанков с дампами `tools/compare_ref.py dump-stream`, а
`VV_TEST_STREAM_E2E=1` поверх — загрузку модели: живая сессия в
`test_stream` (`VV_TEST_STREAM_QUANT`, по умолчанию none) и сессии
WebSocket-сервера по loopback в `test_stream_server` (int4; idle, trickle,
503 при занятом слоте, обрыв клиента). Без этих переменных — SKIP.

Релизная сборка (все архитектуры, статические рантаймы, без тестов):

```bash
scripts/build-release.sh          # Linux
scripts/build-release.ps1         # Windows
scripts/build-macos.sh            # macOS, CPU-путь
```

Кросс-проверка ARM64 (NEON) без Apple-железа: собрать с
`-DCMAKE_TOOLCHAIN_FILE` на aarch64-linux-gnu и запустить
`qemu-aarch64-static -L /usr/aarch64-linux-gnu build-arm64/test_cpu_kernels`.

### Windows

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"

cmake -B build -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89"

cmake --build build --config Release --parallel
ctest --test-dir build --build-config Release --output-on-failure
```

---

## 11. Использование (CLI)

```powershell
# Базовая транскрипция
vv_cli.exe --model ./model_hf --audio recording.wav --output transcript.json

# HTTP-сервер (OpenAI-совместимый, годится как бэкенд GPUStack)
vv_cli serve --model ./model_hf --port 8080 --slots 2 --kv-cache tq4

# Интерактивный цикл с горячей моделью
vv_cli chat --model ./model_hf

# Живая транскрипция с микрофона (или WAV в реальном времени)
vv_cli devices                       # список устройств захвата
vv_cli mic --model ./model_hf --timestamps
vv_cli mic --model ./model_hf --device 1      # индекс, имя или его фрагмент
vv_cli mic --model ./model_hf --from-file meeting.wav

# Квантованный KV и частичный оффлоад
vv_cli --model ./model_hf --audio long.wav --kv-cache tq4 --gpu-layers 20

# С hotwords
vv_cli.exe --model ./model_hf --audio meeting.wav --hotwords "VibeVoice,Azure"

# С ограничением по VRAM (для 12GB карт)
vv_cli.exe --model ./model_hf --audio recording.wav --max-seq-len 8192

# Дамп промежуточных тензоров для сверки с PyTorch (tools/compare_ref.py)
VV_DUMP_DIR=./cdump vv_cli --model ./model_hf --audio recording.wav
```

---

## 12. Метрики качества и производительности

### Целевые метрики (RTX 3080, 10GB)
| Метрика | Цель | Хорошо | Отлично |
|---------|------|--------|---------|
| Prefill (1K tokens) | < 300ms | < 200ms | < 100ms |
| Decode (per token) | < 20ms | < 15ms | < 10ms |
| Throughput (decode) | > 50 tok/s | > 70 tok/s | > 100 tok/s |
| RTF (10 min audio) | < 1.0 | < 0.5 | < 0.3 |
| GPU Memory | < 10 GB | < 9 GB | < 8 GB |
| Model load | < 10s | < 5s | < 2s |
| Speech encoding (10s) | < 500ms | < 200ms | < 50ms |

### Качество (WER — Word Error Rate)
- Должно совпадать с Python reference ± 0.5% WER.
- Диаризация (DER): ± 1% от reference.
- Timestamps: ± 100ms от reference.

---

## 13. Важные технические детали

### Safetensors формат
```
[8 bytes]  header_size (uint64, little-endian)
[N bytes]  header (JSON): {"tensor_name": {"dtype":"U8","shape":[...],"data_offsets":[start,end]}, ...}
[M bytes]  raw tensor data (aligned, contiguous)
```
- Для NF4 слоёв: dtype = "U8" (packed uint8), "F16" или "BF16" для scales.
- Для FP16 слоёв: dtype = "BF16" или "F16".
- Для FP32 слоёв: dtype = "F32".

### bitsandbytes NF4 Memory Layout
```
Для Linear(in=K, out=N) с block_size=64:
  packed_weight:  [N, K/2] uint8     — каждый байт = 2 NF4 значения
  absmax:         [N*K/64] float16   — per-block scale
  quant_state:    metadata (block_size, quant_type, nested quant info)

Double quantization:
  absmax квантуется в FP8 с offset:
  absmax_fp8 + absmax_offset (float32, per superblock)
```

### NF4 Lookup Table (16 values)
```c
static const float NF4_TABLE[16] = {
    -1.0f, -0.6961928f, -0.5250730f, -0.3949338f,
    -0.2844871f, -0.1848489f, -0.0911179f,  0.0f,
     0.0796009f,  0.1609302f,  0.2461123f,  0.3379930f,
     0.4407233f,  0.5626170f,  0.7229568f,  1.0f
};
```

### Causal SConv1d — паддинг (частый источник ошибок)

```
padding_total  = (k - 1) * dilation - (stride - 1)      # НЕ (k - 1)
extra_padding  = out_len * stride - in_len              # нули справа
out_len        = ceil(in_len / stride)
```

Слева кладём `padding_total` нулей, справа `extra_padding`. В стриминге
левый контекст берётся не из нулей, а из хвоста предыдущего чанка
(`k - stride` отсчётов), поэтому результат в точности равен обработке всего
сигнала целиком.

### Conv-VAE Encoder Block (per stage)
```
Each block: norm → mixer(depthwise_conv) → residual + gamma * y
            norm → ffn(linear1 → GELU → linear2) → residual + ffn_gamma * y

  norm  = ConvRMSNorm по каналам, eps = layernorm_eps = 1e-5
  act   = ТОЧНЫЙ GELU (erf), это ACT2FN["gelu"], не tanh-аппроксимация
  ffn   = expansion 4x, bias = conv_bias = true у обоих линейных слоёв

Downsample: strided 1D convolution with stride = ratio[i]

Stage filter progression (encoder_n_filters=32):
  Stage 0: 32 → 32×2=64  (downsample 8x)
  Stage 1: 64 → 64×2=128 (downsample 5x)
  Stage 2: 128 → 128×2=256 (downsample 5x)
  ...etc, doubling filters at each stage
  Final: project to vae_dim
```

### Формат вывода (Rich Transcription)
```json
{
  "text": "Full transcription text...",
  "segments": [
    {
      "speaker": "Speaker 1",
      "start": 0.0,
      "end": 5.24,
      "text": "Hello, welcome to the meeting."
    }
  ],
  "duration": 120.5,
  "language": "en"
}
```

---

## 14. Внешние зависимости (минимальные)

| Зависимость | Версия | Лицензия | Назначение |
|-------------|--------|----------|------------|
| CUDA Toolkit | 12.2+ | NVIDIA EULA | GPU runtime, nvcc, cuBLAS |
| cJSON | 1.7.x | MIT | JSON парсинг (config, output) |

Все остальное — собственная реализация на C.

---

## 15. Streaming для длинного аудио

Из `modeling_vibevoice_asr.py`: аудио > 60s обрабатывается чанками:
```python
segment_samples = int(60.0 * 24000)  # 60s chunks
for start, end in segments:
    chunk = audio[:, start:end]
    acoustic_mean = acoustic_tokenizer.encode(chunk, cache=cache, is_final=is_final).mean
    semantic_mean = semantic_tokenizer.encode(chunk, cache=cache, is_final=is_final).mean
# Concatenate all means, then sample once for acoustic
acoustic_full = concat(acoustic_means)
semantic_full = concat(semantic_means)
```
Tokenizer encoders поддерживают кэш для streaming (VibeVoiceTokenizerStreamingCache).
