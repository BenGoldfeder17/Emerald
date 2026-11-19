<p align="center"><img src="assets/emerald.png" alt="Emerald" width="128"></p>

# Emerald: local AI for the RTX Spark N1X, built on Basalt

Emerald is a local AI inference platform for RTX Spark laptops with the N1X chip, running Windows on Arm. It is built on the
architecture of [Basalt](https://github.com/jesdga95/basalt), Jesús Gulfo's inference engine for NVIDIA Blackwell GPUs, and
serves Qwen3.8-27B entirely on the laptop's GPU through an OpenAI-compatible API.

Basalt itself targets Linux and Windows x64 with compute capability 12.0 (the RTX 50 desktop cards). Emerald brings that
architecture to Windows on Arm and compute capability 12.1, and adds a prebuilt download, two new options and two of
Basalt's Flash-Next options extended to the dense model, all measured on the N1X. The engine, its kernels, the server and
the model packs are Basalt's work; see [Credits](#credits). Emerald is an independent project by
[BenGoldfeder17](https://github.com/BenGoldfeder17): it is not part of Basalt, and Basalt's author does not support it.
The program and its commands keep Basalt's name (`basalt.exe`).

Based on Basalt commit [`63cb68ad`](https://github.com/jesdga95/basalt/commit/63cb68ad471a443811ae171da0395bc3f1fc3ab5) of 2026-11-18. Basalt's own
documentation is kept in [`docs/`](docs/) and applies here except where this page says otherwise.

## Speed against LM Studio

The same RTX Spark N1X laptop, the same prompts, Qwen3.8-27B in both:

| | Emerald | LM Studio |
|---|---|---|
| Essay, temperature 0.6 | 39.1 tok/s | 24.1 tok/s |
| Code, temperature 0 | 70.3 tok/s | 38.4 tok/s |
| First read of an 8,800-token prompt | 2.9 s | 11.3 s |
| Time to first token of a tool call | 0.4 s | 0.7 to 1.4 s |

This compares speed, not answer quality: the two run different 4-bit forms of the model. Emerald ran the NVFP4 pack with
Basalt's drafting and the config below; LM Studio ran the Q4_K_S GGUF on its llama.cpp CUDA runtime. Both had a 32,768-token
context and thinking off, and each figure is the mean of two runs. More in [Measured on the N1X](#measured-on-the-n1x).

## Status

| | |
|---|---|
| Works | Qwen3.8-27B (`Qwen3.8-27B-QUASAR-NVFP4.basalt`, with its DFlash 2 drafter): every weight on the GPU, context up to the model's 262,144 tokens, `basalt serve` with the OpenAI-compatible API, tool calls, the conversation cache. |
| Not available | Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash. They keep their experts in system RAM and compute them with x86 (AVX2, AVX-512) kernels; on ARM64 those kernels are stubs that stop with a message. [Other N1X projects](#other-n1x-projects-built-on-basalt) run Flash-Next. |
| Not tested | Image and video input, more than one GPU, and Basalt's own test suite and quality gate (perplexity and KLD), none of which have been run on ARM64. |

Everything here was built and measured on one laptop. Please report problems with Emerald in this repository, not to Basalt.

## Quick start with the prebuilt download

No compiler is needed. You need an RTX Spark laptop with the N1X chip running Windows 11 on Arm, a current NVIDIA driver
(621.24 was used), an ARM64 Python 3.10 or newer, and about 25 GB of free disk for the model.

1. Download `emerald-v0.1.0-windows-arm64.zip` from [Releases](https://github.com/BenGoldfeder17/Emerald/releases/latest) and
   unzip it. The executables are not code-signed.
2. Add NVIDIA's cuBLAS library, which the engine loads at start. Download the `libcublas` archive from
   <https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-arm64/> (13.8.0.4 was used) and copy
   `cublas64_13.dll` and `cublasLt64_13.dll` from its `bin\arm64` folder into the unzipped folder's `bin`, beside `basalt.exe`.
3. In a command prompt in the unzipped folder, set up the server's Python environment and download the model pack (21 GB)
   from the Basalt author's Hugging Face repository:

   ```bat
   py -3 -m venv .venv
   .venv\Scripts\pip install -r requirements.txt huggingface_hub
   .venv\Scripts\hf download jesdga/Qwen3.8-27B-Basalt Qwen3.8-27B-QUASAR-NVFP4.basalt --local-dir models
   ```

4. Start it:

   ```bat
   bin\basalt.exe serve --config basalt_config.json
   ```

The model loads in about 12 seconds. The API is then at `http://127.0.0.1:8080/v1` and the web UI at `http://127.0.0.1:8080/`.
If Windows reports that `MSVCP140.dll` or `VCRUNTIME140.dll` is missing, install Microsoft's
[Visual C++ Redistributable for ARM64](https://aka.ms/vc14/vc_redist.arm64.exe).

The download carries this `basalt_config.json`, the configuration used on the test laptop:

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

- **GPU memory:** the N1X divides its unified memory between Windows and the GPU. On the test machine (64 GB) the GPU memory
  in use while generating, with the Windows desktop and other running apps included, was about 25 GiB at this 32,768-token
  context and about 32 GiB at the model's full 262,144 (`"max-context": 262144`), both with `"kv": "int8"`.
- **Conversation cache:** the two `conversation-cache` entries keep earlier conversations in RAM and on disk, so returning
  to one does not read it again.
- **Everything else:** Basalt's [configuration](docs/configuration/index.md) and [Qwen3.8-27B](docs/models/qwen38-27b.md) pages.

## Build from source

Besides the above you need:

- Visual Studio 2026 Build Tools with the MSVC ARM64 toolset (19.51 was used) and a Windows SDK, CMake 3.24 or newer and Ninja.
- The CUDA 13 toolkit for `windows-arm64`. NVIDIA publishes it as archives at
  <https://developer.download.nvidia.com/compute/cuda/redist/>; no installer is needed. Release 13.4.2 was used, with these
  packages unpacked into one folder so that their `bin`, `include` and `lib` directories merge: `cuda_nvcc`, `cuda_cudart`,
  `cuda_crt`, `libnvvm`, `libnvptxcompiler`, `cccl`, `libcublas`, `libnvjitlink`, `cuda_nvrtc`, `cuda_nvml_dev`, `cuda_nvtx`
  and `cuda_profiler_api`.

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

The build takes about two minutes and the executables land in `bin\`. Then follow the quick start from step 3; the repository
carries the same `basalt_config.json`, and with `%CUDA_PATH%\bin\arm64` on the path step 2 is not needed. Two build settings
make it an N1X build:

- `-DBASALT_CUDA_ARCH=121` builds for compute capability 12.1 (the default, 120, is Basalt's own target). The start refuses a
  GPU of the other capability.
- ARM64 is detected from the host: the x86 CPU kernels are left out and stubs stand in for them. Only the dense model has
  been built and run, hence `-DBASALT_MODELS=qwen35`.

## Options for the N1X

`--gpu-wait` and `--embedding-gpu` are new. `--prompt-cache-every` and `--relaxed-drafting` are Basalt options that Basalt
has for Flash-Next only; Emerald extends them to the dense model. Each is a command-line flag and, without the dashes, a
key in the config's `engine` section. `gpu-wait`, `embedding-gpu` and `relaxed-drafting` are off unless set. The periodic
checkpoints are on for the dense model; `--prompt-cache-every 0` turns them off.

| Option | What it does | Measured on the N1X |
|---|---|---|
| `--gpu-wait lazy` | The CPU thread that waits for the GPU sleeps through most of each wait and spins only at the end. Other modes: `auto` (default), `spin`, `yield`, `block`. | CPU use while generating falls from about 100% of one core to about 30%, at the same speed. `block` reaches 10% but is about 3.5% slower. |
| `--embedding-gpu` | Keeps the token embedding (2.4 GiB) in GPU memory instead of page-locked system RAM. The N1X's GPU memory is a pool of its own, so this frees RAM for Windows. | Windows available RAM 14.8 to 17.3 GB; same speed, identical greedy output. |
| `--prompt-cache-every N` | The dense model also checkpoints every N fresh prompt tokens (default 8192), so a prompt that shares only part of a turn with an earlier one resumes from the last checkpoint. | A new question about an 8,800-token document already read: 2.7 s to 0.3 s. |
| `--relaxed-drafting P` | For sampled requests, keeps a drafted token when the model gives it a probability of at least P. Greedy requests, added tokens and tool calls stay exact. | An essay with P 0.2 is about 15% faster at temperature 0.6 and 5% at 0.4. This is no longer exact sampling, and its effect on quality has not been measured. |

## Measured on the N1X

One RTX Spark N1X laptop with 64 GB of unified memory, 32,768-token context, `"kv": "int8"`, thinking off, two runs each.
The first column is Emerald with none of its additions switched on, which behaves like Basalt itself. The second adds the periodic checkpoints and
`relaxed-drafting 0.2` for the dense model. The third is LM Studio with the Q4_K_S GGUF of the same model.

| | Basalt's behaviour | With checkpoints and relaxed drafting | LM Studio |
|---|---|---|---|
| Essay, temperature 0.6 | 34.2 tok/s | 39.1 tok/s | 24.1 tok/s |
| Code, temperature 0 | 68.8 tok/s | 70.3 tok/s | 38.4 tok/s |
| First read of an 8,800-token prompt | 2.65 s | 2.88 s | 11.3 s |
| New question about that prompt while it is cached | 2.66 s | 0.33 s | 0.5 to 0.9 s |
| Time to first token of a tool call | 0.39 s | 0.41 s | 0.7 to 1.4 s |

- `embedding-gpu` and `gpu-wait lazy` were added afterwards and did not change these speeds: with all four on, the essay ran
  at 38 to 41 tok/s.
- The GPU has no software power limit on this chip. It drew about 51 W while generating, at 94% utilization.
- Speeds fall by about 5% as the laptop warms up over back-to-back runs.
- At the full 262,144-token context a 212,000-token prompt was read cold in 171 s and answered correctly; generation speed
  was the same as at 32,768.

## Other N1X projects built on Basalt

Emerald is not the only one. Two others were offered to Basalt, which does not take platforms its author cannot test:

- [SolettaSolaris/basalt](https://github.com/SolettaSolaris/basalt/tree/arm64-n1x)
  ([pull request 230](https://github.com/jesdga95/basalt/pull/230)): Windows on Arm64 with NEON CPU kernels, so it also runs
  Qwen3.8-Flash-Next.
- [yanmindi/basalt-arm-sm121](https://github.com/yanmindi/basalt-arm-sm121)
  ([pull request 450](https://github.com/jesdga95/basalt/pull/450)): Windows and Linux on Arm64 with NEON CPU kernels,
  Flash-Next and the 27B, measured on a 128 GB N1X.

Both cover more of Basalt than Emerald does; for Flash-Next or Linux, use one of them. Emerald is for running Qwen3.8-27B on
Windows with the least setup: a prebuilt download (neither of the others had one in November 2026), the N1X options above
and measurements from a 64 GB laptop. It shares no code with either.

## Credits

- **Basalt** is the work of [Jesús Gulfo](https://github.com/jesdga95) and the
  [Basalt contributors](https://github.com/jesdga95/basalt/graphs/contributors): <https://github.com/jesdga95/basalt>,
  [basalt.jgulfo.com](https://basalt.jgulfo.com). The architecture, the CUDA kernels, the drafting, the server and the model
  packs used here are all theirs. If Emerald is useful to you, consider [sponsoring Basalt](https://github.com/sponsors/jesdga95).
- Basalt in turn builds on [Strata](https://github.com/Niko1221/Strata), [Infernix](https://github.com/Wallawalla47/Infernix),
  [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp) and ideas from [NInfer](https://github.com/Neroued/ninfer);
  their licences are in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
- The model is [Qwen](https://huggingface.co/Qwen)'s Qwen3.8-27B in
  [QUASAR-QAT](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)'s NVFP4 form, with
  [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)'s DFlash 2 drafter, under the licences on their cards.
- Emerald itself (bringing Basalt's architecture to Windows on Arm and compute capability 12.1, the options above and the
  prebuilt download) is by [BenGoldfeder17](https://github.com/BenGoldfeder17).

## License

MIT, the same as Basalt. See [LICENSE](LICENSE), which carries Basalt's copyright notice unchanged.
