#!/bin/bash
# Run one rag.py command on a GPU node with a private Ollama server. The arguments are
# passed to rag.py; run and experiment write their results (and plots) to runs/<job id>/.
# The job's output and the Ollama log go to slurm/outputs/. Submit from the assignment
# directory:
#   sbatch slurm/scripts/run_rag.sh experiment indexes --nprobe 4
#   sbatch slurm/scripts/run_rag.sh experiment multihop --n 50
#   sbatch slurm/scripts/run_rag.sh run --index ivf --k 7
#   sbatch slurm/scripts/run_rag.sh query --index flat --question "..."
# Every job runs on the same GPU type (pinned below), so latencies are comparable between
# jobs and with the reference numbers. To use another type (e.g. a Part C experiment
# that needs more GPU memory), override it on the command line and keep those results
# out of comparisons with the pinned type:
#   sbatch --gres=gpu:nvidia_l40s:1 slurm/scripts/run_rag.sh ...
#
#SBATCH -J rag-run
#SBATCH -p gpus
#SBATCH --gres=gpu:geforce_gtx_2080_ti:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
# Relative to the submit directory; slurm/outputs/ must exist or SLURM writes no log
#SBATCH -o slurm/outputs/slurm-%j.out

set -euo pipefail
die() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }

cd "$SLURM_SUBMIT_DIR"
case "${1:-}" in
    experiment|run|query) ;;
    *) die "usage: sbatch slurm/scripts/run_rag.sh {experiment,run,query} [rag.py arguments]" ;;
esac
[ -f rag.py ] || die "submit from the assignment directory (where rag.py is)"
# Data, models, Ollama and Python locations (the course's shared copy, or your own)
source slurm/env.sh

export RAG_LLM_MODEL=${MODEL:-qwen2.5:7b}
# A private Ollama server on a job-unique port: a shared server queues requests and
# corrupts the latency numbers.
export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))
export OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1
export OLLAMA_VULKAN=0
export RAG_OLLAMA_URL=http://$OLLAMA_HOST
# Torch otherwise starts a thread per core of the whole node, and single embeds slow to seconds.
# Leave 2 of the job's cores to the Ollama server: with all 8, embeds took 0.28 s instead of 0.05 s
THREADS=${THREADS:-$(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
# Compute nodes may have no internet: use the cached embedding model, fail fast if missing
export HF_HUB_OFFLINE=1

[ -x "$PY" ] || die "no Python at $PY; run 'uv sync' here on a login node first"
[ -f "$RAG_DATA_DIR/embeddings.npy" ] || die "no data in $RAG_DATA_DIR"
command -v ollama > /dev/null || die "ollama not found on PATH ($PATH)"
$PY -c "from rag_utils import embed_query; embed_query('x')" \
    || die "embedding model not cached; on a login node run:
    source slurm/env.sh && \$PY -c \"from rag_utils import embed_query; embed_query('x')\""

OUT=runs/$SLURM_JOB_ID
OLLAMA_LOG=slurm/outputs/ollama-$SLURM_JOB_ID.log
ollama serve > "$OLLAMA_LOG" 2>&1 &
trap 'kill %1 2>/dev/null' EXIT
for _ in $(seq 60); do
    curl -s "$RAG_OLLAMA_URL/api/version" > /dev/null && break
    sleep 1
done
curl -s "$RAG_OLLAMA_URL/api/version" > /dev/null \
    || die "ollama didn't start in 60s; last log lines:
$(tail -20 "$OLLAMA_LOG")"
ollama list | grep -q "^$RAG_LLM_MODEL " \
    || die "model $RAG_LLM_MODEL not in $OLLAMA_MODELS; 'ollama pull' it on a login node first"

# Load the model now and check that it went to the GPU: a silent fallback to the CPU
# would still run, just 10x slower, and ruin every timing. (rag.py warms up again; cheap.)
curl -s "$RAG_OLLAMA_URL/api/generate" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$RAG_LLM_MODEL\", \"prompt\": \"ok\", \"stream\": false, \"options\": {\"num_predict\": 1}}" > /dev/null
ollama ps | grep -q "100% GPU" \
    || die "model is not fully on the GPU:
$(ollama ps)
last Ollama log lines:
$(tail -20 "$OLLAMA_LOG")"

# Which GPU the job got, saved in each results file's config. nvidia-smi is broken on
# some nodes, so take the name from Ollama's device discovery, else from SLURM.
RAG_GPU=$(grep -oE 'description="[^"]+"' "$OLLAMA_LOG" | head -1 | cut -d'"' -f2)
[ -n "$RAG_GPU" ] || RAG_GPU=$(scontrol show node "$SLURMD_NODENAME" | grep -oE 'Gres=gpu:[^:]+' | cut -d: -f2)
export RAG_GPU
echo "GPU: ${RAG_GPU:-unknown} (node $SLURMD_NODENAME), $THREADS threads, model $RAG_LLM_MODEL, data $RAG_DATA_DIR"
if [ "$1" = query ]; then
    $PY rag.py "$@"
else
    $PY rag.py "$@" --out "$OUT"
fi
