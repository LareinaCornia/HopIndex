# Where the data, the embedding model cache, Ollama and its models, and Python live.
# Sourced by the job script, and by you on a login node before running rag.py there:
#   source slurm/env.sh
# If the course's shared read-only copy exists (RAG_COURSE_DIR), everything comes from
# there; otherwise from your home directory, where the handout's setup steps put it.
# Anything already set in your environment wins over these defaults.
COURSE=${RAG_COURSE_DIR:-/course/cs1390/rag}
if [ -d "$COURSE" ]; then
    export RAG_DATA_DIR=${RAG_DATA_DIR:-$COURSE/rag_data}
    export HF_HOME=${HF_HOME:-$COURSE/hf}
    export OLLAMA_MODELS=${OLLAMA_MODELS:-$COURSE/ollama-models}
    export PATH=$COURSE/ollama/bin:$PATH
    if [ -z "${PY:-}" ]; then
        PY=.venv/bin/python
        [ -x "$COURSE/venv/bin/python" ] && PY=$COURSE/venv/bin/python
    fi
else
    export RAG_DATA_DIR=${RAG_DATA_DIR:-./rag_data}
    export OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/ollama-models}
    export PATH=$HOME/.local/bin:$PATH
    PY=${PY:-.venv/bin/python}
fi
export PY
