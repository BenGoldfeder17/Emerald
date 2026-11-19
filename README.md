<p align="center"><img src="assets/emerald.png" alt="Emerald" width="128"></p>

# Emerald: Basalt for Windows on Arm (RTX Spark N1X)

Emerald is an unofficial port of [Basalt](https://github.com/jesdga95/basalt), the local inference engine for NVIDIA Blackwell
GPUs by Jesús Gulfo, to Windows on Arm and Blackwell compute capability 12.1. With it Basalt runs on RTX Spark laptops with
the N1X chip and serves Qwen3.8-27B entirely on the laptop's GPU. The program and its commands keep Basalt's name (`basalt.exe`).

Upstream Basalt targets Linux and Windows x64 with compute capability 12.0 (the RTX 50 desktop cards). The engine, its kernels,
the server and the model packs are Basalt's work; see [Credits](#credits). This repository adds the platform port, two new
options, and two of Basalt's Flash-Next options extended to the dense model, all measured on the N1X. Port and N1X changes
by [BenGoldfeder17](https://github.com/BenGoldfeder17).

Based on Basalt commit [`63cb68ad`](https://github.com/jesdga95/basalt/commit/63cb68ad471a443811ae171da0395bc3f1fc3ab5) of 2026-11-18. Basalt's own
documentation is kept in [`docs/`](docs/) and applies here except where this page says otherwise.

## Status

| | |
|---|---|
| Works | Qwen3.8-27B (`Qwen3.8-27B-QUASAR-NVFP4.basalt`, with its DFlash 2 drafter): every weight on the GPU, context up to the model's 262,144 tokens, `basalt serve` with the OpenAI-compatible API, tool calls, the conversation cache. |
| Not available | Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash. They keep their experts in system RAM and compute them with x86 (AVX2, AVX-512) kernels; on ARM64 those kernels are stubs that stop with a message. |
| Not tested | Image and video input, more than one GPU, and Basalt's own test suite and quality gate (perplexity and KLD), none of which have been run on ARM64. |

Everything here was built and measured on one laptop. Please report problems with this port in this repository, not upstream.

## Requirements

- An RTX Spark laptop with the N1X chip, running Windows 11 on Arm, with a current NVIDIA driver (621.24 was used).
- Enough GPU memory for the model and its context. The N1X divides its unified memory between Windows and the GPU. On the
  test machine (64 GB) the GPU memory in use while generating, with the Windows desktop and other running apps included,
  was about 25 GiB at a 32,768-token context and about 32 GiB at the full 262,144, both with `"kv": "int8"`.
- Visual Studio 2026 Build Tools with the MSVC ARM64 toolset (19.51 was used) and a Windows SDK, CMake 3.24 or newer, Ninja,
  and an ARM64 Python 3.10 or newer (3.13 was used; every wheel in `requirements.txt` installed prebuilt).
- The CUDA 13 toolkit for `windows-arm64`. NVIDIA publishes it as archives at
  <https://developer.download.nvidia.com/compute/cuda/redist/>; no installer is needed. Release 13.4.2 was used, with these
  packages unpacked into one folder so that their `bin`, `include` and `lib` directories merge: `cuda_nvcc`, `cuda_cudart`,
  `cuda_crt`, `libnvvm`, `libnvptxcompiler`, `cccl`, `libcublas`, `libnvjitlink`, `cuda_nvrtc`, `cuda_nvml_dev`, `cuda_nvtx`
  and `cuda_profiler_api`.

## Build

From a command prompt, with the CUDA folder at `C:\cuda-13.4.2` (change both paths to match your machine):

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64
set "CUDA_PATH=C:\cuda-13.4.2"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\arm64;%PATH%"

git clone https://github.com/BenGoldfeder17/Emerald emerald
cd emerald
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=qwen35 -DBASALT_CUDA_ARCH=121 ^
  "-DCMAKE_CUDA_COMPILER=%CUDA_PATH%/bin/nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_PATH%" ^
  "-DCUDAToolkit_LIBRARY_DIR=%CUDA_PATH%/lib/arm64"
cmake --build build -j 6
```

The build takes about two minutes and the executables land in `bin\`. Two build settings carry the port:

- `-DBASALT_CUDA_ARCH=121` builds for compute capability 12.1 (the default, 120, is upstream's target). The start refuses a
  GPU of the other capability.
- ARM64 is detected from the host: the x86 CPU kernels are left out and stubs stand in for them. Only the dense model has
  been built and run, hence `-DBASALT_MODELS=qwen35`.

## Run

Set up the server's Python environment and download the model pack (21 GB) from the Basalt author's Hugging Face repository
(the `hf` command comes with `pip install huggingface_hub`):

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
hf download jesdga/Qwen3.8-27B-Basalt Qwen3.8-27B-QUASAR-NVFP4.basalt --local-dir models
```

Put the settings in `basalt_config.json`:

```json
{
  "version": 2,
  "models": [{ "model": "models/Qwen3.8-27B-QUASAR-NVFP4.basalt", "gpu": "0", "max-context": 32768, "kv": "int8" }],
  "engine": {
    "vram-reserve-mib": 1024,
    "conversation-cache-mib": 2048,
    "conversation-cache-disk-mib": 8192,
    "embedding-gpu": true,
    "gpu-wait": "lazy",
    "relaxed-drafting": 0.2
  },
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

This is the configuration used on the test laptop. `max-context` can be raised to 262144 when the GPU memory allows it. The
two `conversation-cache` entries keep earlier conversations in RAM and on disk, so returning to one does not read it again.

Start it with the CUDA runtime libraries on the path:

```bat
set "PATH=C:\cuda-13.4.2\bin\arm64;%PATH%"
bin\basalt.exe serve --config basalt_config.json
```

The model loads in about 12 seconds. The API is then at `http://127.0.0.1:8080/v1` and the web UI at `http://127.0.0.1:8080/`.
Every other setting is described in Basalt's [configuration](docs/configuration/index.md) and
[Qwen3.8-27B](docs/models/qwen38-27b.md) pages.

## Options for the N1X

`--gpu-wait` and `--embedding-gpu` are new. `--prompt-cache-every` and `--relaxed-drafting` are Basalt options that upstream
has for Flash-Next only; this port extends them to the dense model. Each is a command-line flag and, without the dashes, a
key in the config's `engine` section. `gpu-wait`, `embedding-gpu` and `relaxed-drafting` are off unless set. The periodic
checkpoints are on for the dense model; `--prompt-cache-every 0` turns them off.

| Option | What it does | Measured on the N1X |
|---|---|---|
| `--gpu-wait lazy` | The CPU thread that waits for the GPU sleeps through most of each wait and spins only at the end. Other modes: `auto` (default), `spin`, `yield`, `block`. | CPU use while generating falls from about 100% of one core to about 30%, at the same speed. `block` reaches 10% but is about 3.5% slower. |
| `--embedding-gpu` | Keeps the token embedding (2.4 GiB) in GPU memory instead of page-locked system RAM. The N1X's GPU memory is a pool of its own, so this frees RAM for Windows. | Windows available RAM 14.8 to 17.3 GB; same speed, identical greedy output. |
| `--prompt-cache-every N` | The dense model also checkpoints every N fresh prompt tokens (default 8192), so a prompt that shares only part of a turn with an earlier one resumes from the last checkpoint. | A new question about an 8,800-token document already read: 2.7 s to 0.3 s. |
| `--relaxed-drafting P` | For sampled requests, keeps a drafted token when the model gives it a probability of at least P. Greedy requests, added tokens and tool calls stay exact. | An essay with P 0.2 is about 15% faster at temperature 0.6 and 5% at 0.4. This is no longer exact sampling, and its effect on quality has not been measured. |

## Measured on the N1X

One RTX Spark N1X laptop with 64 GB of unified memory, 32,768-token context, `"kv": "int8"`, thinking off. The left column is
the port alone, which behaves like upstream Basalt. The right column adds the periodic checkpoints and `relaxed-drafting 0.2`
for the dense model.

| | Port alone | With checkpoints and relaxed drafting |
|---|---|---|
| Essay, temperature 0.6 | 34.2 tok/s | 39.1 tok/s |
| Code, temperature 0 | 68.8 tok/s | 70.3 tok/s |
| First read of an 8,800-token prompt | 2.65 s | 2.88 s |
| New question about that prompt | 2.66 s | 0.33 s |
| Time to first token of a tool call | 0.39 s | 0.41 s |

- `embedding-gpu` and `gpu-wait lazy` were added afterwards and did not change these speeds: with all four on, the essay ran
  at 38 to 41 tok/s.
- The GPU has no software power limit on this chip. It drew about 51 W while generating, at 94% utilization.
- Speeds fall by about 5% as the laptop warms up over back-to-back runs.
- At the full 262,144-token context a 212,000-token prompt was read cold in 171 s and answered correctly; generation speed
  was the same as at 32,768.

## Credits

- **Basalt** is the work of [Jesús Gulfo](https://github.com/jesdga95) and the
  [Basalt contributors](https://github.com/jesdga95/basalt/graphs/contributors): <https://github.com/jesdga95/basalt>,
  [basalt.jgulfo.com](https://basalt.jgulfo.com). The architecture, the CUDA kernels, the drafting, the server and the model
  packs used here are all theirs. If this port is useful to you, consider [sponsoring Basalt](https://github.com/sponsors/jesdga95).
- Basalt in turn builds on [Strata](https://github.com/Niko1221/Strata), [Infernix](https://github.com/Wallawalla47/Infernix),
  [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp) and ideas from [NInfer](https://github.com/Neroued/ninfer);
  their licences are in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
- The model is [Qwen](https://huggingface.co/Qwen)'s Qwen3.8-27B in
  [QUASAR-QAT](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)'s NVFP4 form, with
  [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)'s DFlash 2 drafter, under the licences on their cards.
- The Windows on Arm port, the compute capability 12.1 build and the option work above are by
  [BenGoldfeder17](https://github.com/BenGoldfeder17).

## License

MIT, the same as Basalt. See [LICENSE](LICENSE), which carries Basalt's copyright notice unchanged.
