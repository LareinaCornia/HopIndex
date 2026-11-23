#!/bin/bash
# Open an interactive shell on a GPU node, for debugging. Inside it, run rag.py commands
# through the job script (without sbatch), which starts your private LLM server, sets up
# the environment and prints everything to the terminal:
#   bash slurm/scripts/run_rag.sh query --index flat --question "Who directed Inception?"
# Don't start `ollama serve` yourself: on a shared node it could collide with (or connect
# to) another student's server. Type `exit` when you're done so the GPU goes back to the pool.
#
# Same GPU type as the job script, so timings here match your sbatch jobs. Options:
#   --gpu-type any|<type>  (default geforce_gtx_2080_ti; `sinfo -p gpus -o "%G"` lists types)
#   --time --mem --cpus --gpus

PARTITION="gpus"
GPU_TYPE="geforce_gtx_2080_ti"
NGPUS="1"
CPUS="8"
MEM="32G"
TIME="2:00:00"   # enough for one experiment; the GPU stays yours until you exit or this runs out

while [[ $# -gt 0 ]]; do
  case $1 in
    --time) TIME="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --gpus)
      if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: --gpus must be a positive integer, got '$2'"
        exit 1
      fi
      NGPUS="$2"
      shift 2
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "$GPU_TYPE" == "any" ]]; then GRES="gpu:$NGPUS"; else GRES="gpu:$GPU_TYPE:$NGPUS"; fi
mkdir -p slurm/outputs

echo "Requesting $GRES, $CPUS CPUs, $MEM, for up to $TIME..."
echo "Once on the node, run rag.py through the job script, e.g.:"
echo "  bash slurm/scripts/run_rag.sh query --index flat --question \"Who directed Inception?\""
srun \
  --partition=$PARTITION \
  --gres=$GRES \
  --time="$TIME" \
  --mem="$MEM" \
  --cpus-per-task="$CPUS" \
  --pty \
  $SHELL
