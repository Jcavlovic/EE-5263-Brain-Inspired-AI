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

### Run

Cross-platform (works on Windows, Linux, macOS):

```bash
python src/main.py               # full sweep: epochs ∈ {25, 100, 200} — hours on CPU
python src/main.py --quick       # caps epoch-sweep at 25 epochs — ~25 min on CPU
python src/main.py --only hidden_layers sparsity   # subset of sweeps
```

On Windows, `--workers 0` (the default) is recommended; raise it on Linux/
macOS for faster image loading. GPU is used automatically when
`torch.cuda.is_available()` is true, otherwise the code falls back to CPU.

Outputs land in `results/`:

- `sweep_hidden_layers.png`, `sweep_neurons_per_layer.png`, `sweep_sparsity.png`,
  `sweep_learning_rate.png`, `sweep_epochs.png`, `sweep_batch_size.png`
- `summary.json` — defaults used plus every run's per-epoch stats,
  final/best accuracy, and wall time.
