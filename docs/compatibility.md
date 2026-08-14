# Compatibility

## Python / OS

| Python | Ubuntu | macOS | Windows |
|--------|--------|-------|---------|
| 3.9 | ✓ | ✓ | ✓ |
| 3.10 | ✓ | ✓ | ✓ |
| 3.11 | ✓ | ✓ | ✓ |
| 3.12 | ✓ | ✓ | ✓ |

Core extraction (no ML extras) is tested on all three platforms in CI. Local vision model extras are developed and tested on Linux with NVIDIA GPUs.

## Local vision model extras

| Extra | transformers | torch | CUDA | Notes |
|---|---|---|---|---|
| `[qwen2vl]` | ≥4.49, <6.0 | ≥2.1, <3.0 | 11.8 / 12.1 / 12.4 | Recommended for document/chart understanding. ≥16 GB VRAM in BF16; use `load_in_4bit=True` for 8–12 GB cards. |
| `[llama]` | ≥4.45, <6.0 | ≥2.1, <3.0 | 11.8 / 12.1 / 12.4 | Llama 3.2 Vision 11B/90B. Same VRAM requirements as Qwen2.5-VL. |
| `[smolvlm]` | ≥4.49, <6.0 | ≥2.1, <3.0 | CPU / any | 2.2B parameters; runs on CPU. No `trust_remote_code` required. |

## CUDA wheel selection

```bash
# CUDA 11.8
pip install "multixtract[qwen2vl]" --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install "multixtract[qwen2vl]" --extra-index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install "multixtract[qwen2vl]" --extra-index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install "multixtract[smolvlm]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Replace `[qwen2vl]` with `[llama]` or `[smolvlm]` as needed.
