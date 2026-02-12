# Skill: BPE Text Tokenizer

## Purpose
Pure C BPE tokenizer for Qwen2.5 vocab (152064 tokens).
Used for text prompts and decoding output tokens.

## NOTE
Tokenizer files not in 4-bit repo. Fetch from microsoft/VibeVoice-ASR
or Qwen2.5-7B base model.

## Key API
```c
vv_status_t vv_tokenizer_load(const char* path, vv_tokenizer_t** out);
vv_status_t vv_tokenizer_encode(const vv_tokenizer_t* tok, const char* text,
                                 int32_t** ids, int* n_tokens);
vv_status_t vv_tokenizer_decode(const vv_tokenizer_t* tok,
                                 const int32_t* ids, int n_tokens, char** text);
int vv_tokenizer_special_id(const vv_tokenizer_t* tok, const char* name);
```

## Special Tokens for VibeVoice-ASR
- `<|startoftranscript|>`, `<|endoftranscript|>`
- `<|speaker_N|>`, `<|timestamp_X.XX|>`
- `<|hotwords|>`, `<|nospeech|>`
