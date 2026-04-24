# EE-5263-Brain-Inspired-AI
Repo for UTSA EE-5263 Brain Inspired AI Coursework

## Rate-based gender classifier on LFW

A rate-coded MLP (stacked sigmoids, supervised learning) that classifies the
gender of each Labeled Faces in the Wild image. `src/main.py` runs a
one-at-a-time parameter sweep (hidden layers, neurons/layer, sparsity,
learning rate, epochs, batch size) and writes a PNG per parameter plus
`summary.json` into `results/`.

Gender labels are derived from each identity's first name via the
`gender-guesser` library; identities whose first name is ambiguous are
skipped. Train and test are split at the identity level so the same person
never appears in both sets; each split is then balanced by downsampling the
majority class.

### Setup — Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-project.txt
```

### Setup — Windows 11 (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-project.txt
```

(If script execution is blocked:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)

### Dataset layout

The LFW images must live under:

```
data/LFW/lfw-deepfunneled/lfw-deepfunneled/<Person_Name>/<Person_Name>_NNNN.jpg
```

Pass a different location via `--lfw-root <path>` if needed.

### Run — one-at-a-time sweeps (6 plots)

```bash
python src/main.py               # full sweep: epochs ∈ {25, 100, 200} — hours on CPU
python src/main.py --quick       # caps epoch-sweep at 25 epochs — ~25 min on CPU
python src/main.py --only hidden_layers sparsity   # subset of sweeps
python src/main.py --preload     # same runs but cache all images in RAM first — ~15× faster
```

Outputs in `results/`:
- `sweep_<parameter>.png` (six of them), `summary.json`.

### Run — full cross-product (1,152 configs)

```bash
python src/main.py --cross-product        # 4 × 4 × 4 × 3 × 3 × 2 = 1,152 runs
```

Implies `--preload`. Streams one JSON line per completed run into
`results/cross_results.jsonl`, so interrupting and rerunning the same command
skips already-completed configs. Estimated wall time at full depth (all
epoch counts including 200): **~5 h on an RTX 3070**, ~24 h on a modern
multi-core CPU.

Outputs in `results/`:
- `cross_results.jsonl` — one record per run (config + best/final acc + wall time). Resumable.
- `cross_marginals.png` — mean best test accuracy per value of each axis.
- `cross_heatmaps.png` — grid of `hidden_layers × neurons_per_layer` heatmaps, one panel per `(learning_rate, epochs, batch_size)` triple, averaged over sparsity.
- `cross_top_configs.txt` — top-20 configs by best test accuracy.

### Notes

- GPU used automatically when `torch.cuda.is_available()`; CPU fallback otherwise.
- With `--preload` the cached tensor is ~90 MB at the default 64×64 grayscale size; on GPU it lives in VRAM so `__getitem__` returns device-local views.
- On Windows, `--workers 0` (the default) is the safest choice because of spawn-mode pickling. Preloaded data doesn't use workers at all.
