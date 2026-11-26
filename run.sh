# Historical script from the first batch of runs (seeds 0-2), kept for audit only.
# It rewrites environment.txt and re-clones ADBench; do not use it for setup. See README.
set -euo pipefail
cd -- "$(dirname -- "$0")"
trap 'status=$?; printf "%s\n" "$status" > exit-code' EXIT
export PATH="$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo PHASE_SETUP
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python --version
uv pip freeze --python .venv/bin/python > environment.txt
git clone --depth 1 --filter=blob:none --sparse https://github.com/Minqi824/ADBench data_src/adbench_repo
git -C data_src/adbench_repo sparse-checkout set adbench/datasets/Classical adbench/datasets/CV_by_ResNet18 adbench/datasets/NLP_by_BERT
git -C data_src/adbench_repo rev-parse HEAD > adbench-commit.txt
echo PHASE_SMOKE
nice -n 10 .venv/bin/python -m anoddae_repro.runner --exp sweepT_50 impr_trim0.1_T50 --setting unsup --seeds 0 --protocol dte2 --datasets 6_cardio --workers 1 --tag _full
.venv/bin/python - <<'PY'
import json
from pathlib import Path
for exp in ("sweepT_50", "impr_trim0.1_T50"):
    path = Path("results") / f"{exp}_full.jsonl"
    record = json.loads(path.read_text().splitlines()[-1])
    assert "error" not in record and record["file"] == "6_cardio", record
PY
echo PHASE_FULL
nice -n 10 .venv/bin/python -m anoddae_repro.runner --exp sweepT_50 impr_trim0.1_T50 --setting unsup --seeds 0 1 2 --protocol dte2 --workers 4 --tag _full
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from anoddae_repro.data import benchmark_datasets, family_members
expected = {(member, seed) for dataset in benchmark_datasets() for member in family_members(dataset) for seed in (0, 1, 2)}
for exp in ("sweepT_50", "impr_trim0.1_T50"):
    path = Path("results") / f"{exp}_full.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    done = {(row["file"], row["seed"]) for row in rows if "error" not in row}
    if missing := expected - done:
        raise SystemExit(f"{exp}: {len(missing)} incomplete jobs")
    print(f"{exp}: {len(done)} successful jobs")
PY
echo PHASE_COMPLETE
