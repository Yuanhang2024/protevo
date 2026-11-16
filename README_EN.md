# protevo — Universal Protein Directed-Evolution Agent Framework

**中文版: [README.md](README.md)**

A protein-agnostic directed-evolution framework built around an LLM tool-calling agent: swap the CSV
and you swap the protein. GB1 (protein G B1 domain, IgG-Fc binding) is the framework's first fully
documented case study (243-configuration predictor ablation, 2×2 factorial experiment, double-blind
statistics); three further real-world ProteinGym datasets — GFP / ubiquitin / TEM-1 (indels) — were
replayed cross-protein with zero code changes.

## One-Command Launch

```bash
# Windows: double-click run.bat, or run from the command line
run.bat

# Linux / macOS
sh run.sh
```

The script automatically: detects Python 3.10+ → creates .venv → installs dependencies (a few
minutes the first time, skipped afterwards) → drops you into the interactive command-line workbench.
The environment variable `PROTEVO_PY` can point to a specific interpreter path. The TUI is pure
standard library with one entry point on all three platforms; verified on Windows, while Linux/macOS
share the same fully cross-platform dependency set and workflow.

## Quick Start

```python
from protevo.protein_config import configure_protein
from protevo.agent_v2 import run_agent_round

config = configure_protein("data/external/ubiq_roscoe_2014/dms.csv",
                           reference_sequence=<WT>)   # auto-detects format/positions/direction and selects a predictor
config.predict_fitness(["I3V", "K6R"])                # native format in, native scale out
```

```bash
# Any dataset straight through (--csv mode, no code changes needed)
python run_cross_protein.py --mode notation --csv data/external/gfp_sarkisyan/dms.csv --name gfp

# Unsure about local hardware/environment? Let an LLM help configure it
# (without LLM credentials it prints a deterministic rule-based plan)
python -m protevo.setup_assistant data/external/blat_ecolx_gonzalez_2019/dms.csv --ask "How should this machine be configured?"
```

### One command for a full multi-round virtual directed evolution

```bash
python -m protevo.evolve data/external/ubiq_roscoe_2014/dms.csv --rounds 2 --reference <WT>
# Results land in results/evolve/<dataset>_<timestamp>/: per_round.csv / summary.json / agent_round*.jsonl
```

Virtual evolution = the CSV's full fitness column serves as a lookup-back-fill oracle: each round's
submitted variants are matched verbatim and back-filled with their "measured" scores, so the complete
multi-round loop can be replayed without wet experiments. For a real wet-lab campaign, call
`run_agent_round` round by round and feed measured data in manually between rounds.

Two predictor protocols (top-level `predictor_protocol` in summary.json): the default `fixed` — the
predictor is fit once on the initial training subset and never retrained across rounds;
`run_evolution(..., retrain_each_round=True)` selects `refit_per_round` — before each round the
predictor is re-selected and re-trained on the accumulated known set (initial training + all prior
back-filled feedback; each round fits only on the initial training set and this method's own prior
feedback), and each round's `predictor_info` is recorded in the rounds detail.

## Command-Line Workbench (TUI)

`python -m protevo.tui` starts the command-driven workbench with one command (stdlib `cmd.Cmd`, zero
new interactive dependencies; a little ANSI color, disabled automatically on non-terminals).
**Full coverage** (data & config / experiments / visualization / replay & export — four panels →
four command groups); rerun `setup` anytime for the wizard, or pass `--no-wizard` to skip it.

On first launch the package-root `.env` is auto-detected: if credentials are incomplete and you are
in a real terminal, a three-step wizard runs — ① Base URL (OpenAI official / DeepSeek pre-filled /
custom compatible endpoint) ② API key (silent getpass input) ③ Model name (an endpoint-appropriate
suggested default) — and after a confirmation summary writes `.env` (only appending/updating those
three keys; every other line is left untouched), then points to the `doctor` llm_credentials check
for review. Every step is skippable: **the CPU predictor path needs no credentials** (datasets /
open / predict / plot / export are fully offline); only run (agent evolution) and ask (LLM advisor)
need them. When stdin is not a tty (pipes/tests) it never blocks: it prints one credential notice
and enters the REPL directly.

| Command | Description |
|---|---|
| `datasets` | List the 4 built-in example datasets (gb1 / ubiq / gfp / blat) |
| `open <csv\|builtin> [--ref WT] [--esm] [--pool recommended\|full] [--hrb higher\|lower] [--lang zh\|en]` | Configure a dataset and print the config card (format / positions / reference source incl. unreliability warning / direction / selected predictor with val ρ / all candidates) |
| `info` / `status` | Current config card / session overview (data · predictor · credentials · last run) |
| `run [--rounds 3] [--per-round 10] [--budget 40] [--retrain]` | Multi-round virtual evolution; one near-real-time summary line per round (round / submitted / back-filled mean / cumulative best / budget left), plus a small trajectory table at the end |
| `model [name]` | View/set this session's LLM model name (use setup to persist) |
| `lang [zh\|en]` | View/switch the agent prompt language (only the system prompt sent to the LLM changes; the TUI interface stays Chinese) |
| `plot <scatter\|trajectory\|heatmap\|length\|budget>` | matplotlib renders a PNG into the output directory + a one-line ASCII summary in the terminal (auto-degrades to ASCII-only when matplotlib is missing); heatmap targets substitution libraries, length targets indel libraries, and each guides you to the other when misused |
| `log [round]` | Replay agent_round*.jsonl: per-round summaries by default; with a round number, every record's kind/tool/arguments (incl. submitted variants + reasons) |
| `export [--out DIR]` | Export per_round.csv / summary.json / all applicable plots and print the manifest |
| `plan` / `doctor [--execute]` | Data probing + rule-based plan / environment check-up (with optional auto-fix) |
| `ask <question>` | LLM advisor (prints a degraded notice without credentials) |
| `predict SEQ...` | Score with the current predictor (native format in, native scale out) |
| `setup` / `help` / `quit` | Re-run the three-step wizard / command table / exit (EOF works too) |

The agent prompt language is selectable: `open <csv> --lang en` to switch agent prompt language (or
`lang [zh|en]` to toggle anytime; only the system prompt sent to the LLM changes — the TUI interface
text stays Chinese).

Example session (`open → run → plot → export`):

```
protevo> open ubiq                       # built-in ubiquitin library: format/positions/direction auto-detected, predictor auto-selected
protevo (ubiq|RF ρ=0.78)> run --rounds 2 # one summary line per round + a small trajectory table (needs .env credentials)
protevo (ubiq|RF ρ=0.78)> plot trajectory# trajectory PNG into the latest output directory + ASCII summary
protevo (ubiq|RF ρ=0.78)> log 1          # replay round 1 tool calls and submission reasons record by record
protevo (ubiq|RF ρ=0.78)> export         # per_round.csv / summary.json / all applicable plots bundled
```

## Supported Variant Formats (auto-detected)

| Format | Example | Notes |
|---|---|---|
| `full` | full-length sequence CSV | mutable positions detected by full scan |
| `mutation_string` | `VDGV` | GB1-style mutation string |
| `mutation_notation` | `V39D,G40E` / `K3R:V55A` | ProteinGym/MaveDB notation (colon-separated natively supported) |
| `indel_notation` | `V39del` / `V39_G40insAA` / `V39dup` | HGVS subset; sequence-level predictors (composition / ESM-2) |

## Cross-Protein Validation (`results/cross_protein/summary.json`)

| Protein | Library structure | Auto-selected | val Spearman |
|---|---|---|---|
| GB1 (Wu 2016, 149,361) | 4-position substitution enumeration | one-hot + RF | 0.6554 (50k training) |
| GFP (Sarkisyan 2016, 51,714) | 233-position substitutions | one-hot + Ridge | 0.8652 (esm2_mlp 0.7765) |
| Ubiquitin RL40A (Roscoe 2014, 1,380) | 75-position single substitutions | one-hot + RF | 0.7849 (esm2_mlp 0.6357) |
| TEM-1 BLAT (Gonzalez 2019, 4,751) | single-site indels | **ESM-2 embeddings + MLP** | 0.7552 (esm2_ridge 0.6568; composition baseline 0.0323) |

The selection protocol picked correctly on all four library structures (the candidate pool already
includes esm2_mlp — second overall in the 243-configuration ablation — which shares one embedding
matrix with esm2_ridge so embeddings are computed only once): on substitution libraries the one-hot
family won three times in a row — esm2_mlp reached only 0.7765/0.6357 on GFP/ubiquitin and still
lost to the one-hot family; on the indel library ESM turned the tables and the best head upgraded
from Ridge to MLP — exactly where fixed positional encodings fail is where PLM embeddings shine.

## Structure

```
protevo/                generic framework layer + GB1 experiment infrastructure layer (see the two-layer positioning in protevo/__init__.py)
  protein_config.py      one-line configuration entry (four formats / direction normalization / predictor selection / optional ESM)
  variant_notation.py    indel-notation core (parse / validate / reconstruct)
  prompt_builder.py      config-driven system-prompt template (zh/en)
  tools.py               ToolBox + build_tool_schemas(protein_config)
  agent_v2.py            tool-calling loop (protein_config injection, byte-compatible with the legacy GB1 path)
  embedding_service.py   on-demand ESM-2 embeddings (LRU + on-disk cache, lazy torch, device=auto hardware adaptation)
  setup_assistant.py     LLM-assisted environment setup (deterministic probing + constrained menu + guard validation)
  tui.py                 command-line workbench (cmd.Cmd, first-run wizard + full four-panel functionality, replaces the graphical plugin)
tests/                   536 tests (520 passed + 16 skipped: GPU-gated and optional-data tests); the interactive interface is the built-in TUI, zero external plugin dependencies
data/external/           three ProteinGym datasets (MIT, provenance in each meta.json)
docs/release_audit.md    pre-release audit report (dataset pinning documented item by item)
results/                 cross-protein results + GB1 ablation/factorial/statistics summaries (bulky raw logs not shipped)
```

> The package was renamed from `gb1` to `protevo` at v0.2.0 (this package). The GB1-specific
> experiment infrastructure (schema/data/agent/research_session) is frozen as-is for reproducibility;
> the full inventory is in `docs/release_audit.md`. Interactive capability is provided by the
> built-in TUI (`python -m protevo.tui`), with zero external plugin dependencies.

## Environment & Testing

```bash
pip install -r requirements.txt      # full CPU functionality
python -m pytest tests -q            # expected all green (GPU-gated tests skip automatically)
```

The LLM agent and setup_assistant's LLM advisor layer need a `.env`: `OPENAI_API_KEY` /
`OPENAI_MODEL` (optional `OPENAI_BASE_URL`; any OpenAI-compatible endpoint works). Without
credentials, setup_assistant degrades to a deterministic rule-based plan.

### Hardware probe (no error even without torch)

```python
from protevo.embedding_service import hardware_probe
print(hardware_probe())
# {'torch': True, 'cuda': True, 'backend': 'CUDA 12.8',
#  'device_name': 'NVIDIA GeForce RTX 5090 Laptop GPU', 'vram_gb': 23.9, 'transformers': True}
```

### ESM-2 embedding path (use_esm=True, optional enhancement)

`configure_protein(..., use_esm=True)` adds ESM-2 embedding predictors to the candidate pool:
esm2_ridge (Ridge) and esm2_mlp (MLP, same spec as the runner-up of the 243-configuration ablation;
the two candidates share one embedding matrix, so embeddings are computed only once; the optional
`esm_cache_dir` enables on-disk embedding reuse — `run_cross_protein.py --embed-cache` enables it by
default). If torch/transformers/checkpoint are unavailable, it prints an explanation and skips them
gracefully without affecting the CPU path. Three steps to get ready:

**1. Install torch for your platform (transformers also required)**

```bash
# CPU (no discrete GPU, or just wiring up the pipeline)
pip install torch transformers

# NVIDIA CUDA (cu126 is just an example — pick by your driver version, see https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers

# AMD ROCm (under ROCm, PyTorch maps hip devices to cuda; the device name stays "cuda", zero code changes)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install transformers
```

**2. Prepare the ESM-2 checkpoint**

Automatic (recommended) — the first call downloads ~2.5 GB to the default cache
`~/.cache/protevo/esm2_t33_650M`; if already present it is reused directly with no repeat download:

```python
from protevo.embedding_service import ensure_esm_checkpoint
from protevo.protein_config import configure_protein

model_path = ensure_esm_checkpoint()
config = configure_protein("gb1.csv", use_esm=True, esm_model_path=model_path)
```

On slow networks in mainland China, use the hf-mirror mirror (huggingface_hub natively honors the
`HF_ENDPOINT` environment variable):

```bash
set HF_ENDPOINT=https://hf-mirror.com        # Linux/macOS: export HF_ENDPOINT=https://hf-mirror.com
```

Or download manually and pass the directory to `esm_model_path` (both direct and mirror forms):

```bash
huggingface-cli download facebook/esm2_t33_650M_UR50D --local-dir D:/models/esm2_t33_650M_UR50D
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download facebook/esm2_t33_650M_UR50D --local-dir D:/models/esm2_t33_650M_UR50D
```

**3. Enable ESM and run**

```python
config = configure_protein("my_data.csv", use_esm=True)
# esm_device defaults to "auto": use the GPU automatically when CUDA is detected (incl. ROCm,
# where hip is mapped to cuda), otherwise fall back to CPU; you can also pass esm_device="cuda" /
# "cpu" explicitly (an explicit "cuda" on a GPU-less machine raises a clear error)
```

### LLM-Assisted Environment Setup (setup_assistant)

`python -m protevo.setup_assistant <data.csv> [--reference WT] [--ask question] [--execute]`:
probes hardware (torch/CUDA/ROCm/VRAM/checkpoint) and data (format/uniform-library guardrails/
indels); the LLM then chooses configuration from a **constrained menu** (device / protein model
650M|8M / torch install command / checkpoint action), and a guard layer validates every field —
choices contradicting hardware or data facts (e.g. cuda without a GPU, 650M with little VRAM, ESM on
mutation-string data) are automatically overwritten and logged. Scientific model selection (which of
RF/Ridge/MLP/ESM is best) is always decided by the validation-Spearman protocol; the LLM only
configures the environment and never picks the model.

The selection candidate pool can be opened up (priors only recommend, never block): the default
`predictor_pool="recommended"` is the internally and literature-calibrated recommended set;
`predictor_pool="full"` opens every representation-feasible combination for that format (e.g.
onehot_mlp), `extra_candidates=["onehot_mlp"]` appends fine-grained, and
`run_cross_protein.py --pool full` passes the same choice through. Representation-infeasible
combinations (one-hot on indels; ESM/composition on mutation strings/notation) stay excluded, with
reasons recorded in `predictor_info["pool"]["excluded"]` — that is a representation fact, not a
prior.

- **Configured short-circuit (idempotent)**: after `--doctor` passes or `--execute` succeeds,
  `~/.cache/protevo/setup_state.json` is written; subsequent default runs just print the "already
  configured" summary and exit (no LLM call is made).
- **`--doctor` environment check-up**: item-by-item checks of Python / core dependencies / torch
  build and platform match / transformers / checkpoint weights / embedding and predictor smoke tests
  / LLM credentials, each with a fix command; with `--execute` a missing checkpoint is automatically
  downloaded and the checks rerun.
- **Fingerprint auto-invalidation**: the state file records a hardware fingerprint (backend/cuda/
  device name/VRAM ±1 GB tolerance); after a card or driver change the machine is automatically
  treated as unconfigured and the normal flow reruns (`--force` forces reconfiguration manually).

## License

MIT (see LICENSE). Dataset provenance and licenses are in `data/external/*/meta.json`
(ProteinGym v1.3, Zenodo 10.5281/zenodo.15293562).
