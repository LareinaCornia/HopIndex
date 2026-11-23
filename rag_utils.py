import collections
import functools
import json
import os
import random
import re
import string
import time
import urllib.error
import urllib.request

import numpy as np

# The corpus and the embedding model live in the course's shared read-only copy. On GPU
# nodes the job script's slurm/env.sh sets the paths. Anywhere else, use ./rag_data if
# present, else the course copy, and the course's embedding model cache if it exists.
COURSE_DIR = "/course/cs1390/rag"
COURSE_DATA_DIR = os.path.join(COURSE_DIR, "rag_data")
DATA_DIR = os.environ.get("RAG_DATA_DIR") or (
    "./rag_data" if os.path.exists("./rag_data") else COURSE_DATA_DIR
)
# Must be set before anything imports huggingface_hub (embed_query() imports it lazily)
if not os.environ.get("HF_HOME") and os.path.isdir(os.path.join(COURSE_DIR, "hf")):
    os.environ["HF_HOME"] = os.path.join(COURSE_DIR, "hf")
SEED = 43

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge expects this prefix on queries (not passages); without it recall silently drops
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.environ.get("RAG_OLLAMA_URL", "http://localhost:11434")


class RAGData:
    """
    The HotpotQA corpus and questions.

    Fields:
    titles, texts: parallel lists, one entry per passage
    title_ids: dict from a title to its corpus id (titles are unique)
    embeddings: (n_passages, 384) L2-normalized float32 array, same order as titles/texts
    qa: list of question dicts with keys qid, question, answer, type, gold_titles
    """

    def __init__(self, data_dir=DATA_DIR):
        if not os.path.exists(os.path.join(data_dir, "embeddings.npy")):
            raise FileNotFoundError(
                f"no corpus in {data_dir}/. On the cluster it is in {COURSE_DATA_DIR}. "
                f"Elsewhere, set RAG_DATA_DIR to a copy of it."
            )

        self.titles, self.texts = [], []
        with open(os.path.join(data_dir, "titles.jsonl")) as f:
            for line in f:
                row = json.loads(line)
                self.titles.append(row["title"])
                self.texts.append(row["text"])
        self.embeddings = np.load(os.path.join(data_dir, "embeddings.npy"))
        assert len(self.titles) == self.embeddings.shape[0]
        self.title_ids = {title: i for i, title in enumerate(self.titles)}

        with open(os.path.join(data_dir, "qa.jsonl")) as f:
            self.qa = [json.loads(line) for line in f]
        for i, row in enumerate(self.qa):
            row["qid"] = i
            # supporting_facts_titles repeats a title once per supporting sentence
            row["gold_titles"] = sorted(set(row["supporting_facts_titles"]))

    def sample_questions(self, n, seed=SEED):
        """
        Stratified (bridge/comparison) sample. The same (n, seed) always gives the same
        questions, and a smaller n is a prefix of a larger one, so runs stay paired.
        """
        rng = random.Random(seed)
        by_type = {}
        for row in self.qa:
            by_type.setdefault(row["type"], []).append(row)
        for group in by_type.values():
            rng.shuffle(group)
        order, taken = [], {t: 0 for t in by_type}
        while len(order) < len(self.qa):
            t = min(by_type, key=lambda t: (taken[t] + 1) / len(by_type[t]))
            if taken[t] < len(by_type[t]):
                order.append(by_type[t][taken[t]])
            taken[t] += 1
        return order[:n]


@functools.lru_cache(maxsize=None)
def _embed_model():
    # sentence_transformers takes several seconds to import
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed_query(question):
    """
    Embed a question for retrieval. The first call loads the model (slow); later calls
    reuse it.

    Returns:
    A (384,) L2-normalized float32 vector, comparable with RAGData.embeddings
    """
    return _embed_model().encode(QUERY_PREFIX + question, normalize_embeddings=True)


class OllamaClient:
    """
    Talks to an Ollama server. generate() returns the model's text plus Ollama's own
    timing breakdown, so callers can split LLM time into prefill / decode / overhead.
    """

    def __init__(self, model=LLM_MODEL, url=OLLAMA_URL):
        self.model = model
        self.url = url

    def generate(self, prompt, num_predict=32):
        """
        Parameters:
        prompt: the full prompt string
        num_predict: maximum number of tokens to generate

        Returns:
        (text, stats):
            - text: the model's reply, stripped
            - stats: dict with prompt_tokens, output_tokens, and prefill_s, decode_s,
              load_s, server_s (Ollama's total time for the request), all in seconds
        """
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                # 4096 should be sufficiently long to not truncate any of our prompts
                "options": {
                    "temperature": 0.0,
                    "seed": 0,
                    "num_predict": num_predict,
                    "num_ctx": 4096,
                    "num_thread": 2,
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(3):  # Ollama occasionally returns a transient 500
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    body = json.loads(resp.read())
                break
            except urllib.error.HTTPError:
                if attempt == 2:
                    raise
                time.sleep(5)
            except urllib.error.URLError as e:
                raise RuntimeError(
                    f"Could not reach Ollama at {self.url} ({e.reason}). Is `ollama serve` "
                    f"running, and has `ollama pull {self.model}` been done?"
                ) from e
        ns = 1e-9
        stats = {
            "prompt_tokens": body.get("prompt_eval_count", 0),
            "output_tokens": body.get("eval_count", 0),
            "prefill_s": body.get("prompt_eval_duration", 0) * ns,
            "decode_s": body.get("eval_duration", 0) * ns,
            "load_s": body.get("load_duration", 0) * ns,
            "server_s": body.get("total_duration", 0) * ns,
        }
        return body.get("response", "").strip(), stats

    def warmup(self):
        """Load the model so the first timed call doesn't pay for it."""
        self.generate("Reply with the single word: ok", num_predict=2)

    def unload(self):
        """
        Unload the model, which also drops the server's prompt cache; warmup() reloads it.
        Between runs over the same questions, cached prompt prefixes would otherwise make
        the later run's prefill look faster than it is.
        """
        payload = json.dumps({"model": self.model, "keep_alive": 0}).encode()
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()


class FakeLLM:
    """
    Stand-in for OllamaClient that needs no server: always answers "yes", with made-up
    but consistently shaped stats. For testing the pipeline, not for measurements.
    """

    model = "fake"

    def generate(self, prompt, num_predict=32):
        prompt_tokens = len(prompt) // 4
        stats = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": 1,
            "prefill_s": prompt_tokens * 1e-5,
            "decode_s": 0.01,
            "load_s": 0.0,
        }
        stats["server_s"] = stats["prefill_s"] + stats["decode_s"] + 0.005
        time.sleep(stats["server_s"])
        return "yes", stats

    def warmup(self):
        pass

    def unload(self):
        pass


def make_llm():
    """FakeLLM if RAG_FAKE_LLM=1 is set, otherwise a real OllamaClient."""
    return FakeLLM() if os.environ.get("RAG_FAKE_LLM") == "1" else OllamaClient()


# Given pipeline pieces: retrieval, the answer prompt, and the oracle.


def retrieve(index, query_vec, k):
    """
    Find the k passages closest to the query using one of your Part 1 indexes.

    Parameters:
    index: a trained MyIndexFlat, MyIndexIVF or MyIndexPQ over the corpus embeddings
    query_vec: the embedded question, a (384,) numpy array
    k: the number of passages to retrieve

    Returns:
    A list of k corpus ids (plain Python ints), nearest first
    """
    _, ids = index.search(query_vec, k)
    return [int(i) for i in ids]


def build_prompt(question, passages):
    """
    Build the prompt that asks the LLM to answer the question from the passages.

    Parameters:
    question: the question string
    passages: a list of (title, text) pairs

    Returns:
    The prompt string
    """
    context = "\n\n".join(f"[{title}]\n{text}" for title, text in passages)
    return (
        "Answer the question using ONLY the context below. "
        "Give the SHORTEST possible answer: a name, date, number, or short phrase. "
        "For yes/no questions answer only yes or no. "
        "Do not answer in a full sentence. Do not explain.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# Pass k = 2 to get gold only oracle
def oracle(row, index, k, data, llm):
    """
    Answer with both gold passages guaranteed in the context: the upper bound on what
    better retrieval could buy. The gold passages are padded with the top retrieved
    non-gold ones up to k, in a fixed per-question shuffled order.

    Parameters:
    row: the question dict (from RAGData.qa)
    index, k, data, llm: as in rag_pipeline.answer()

    Returns:
    The same dict as rag_pipeline.answer()
    """
    stages = {"embed": 0.0, "search": 0.0, "generate": 0.0}

    t0 = time.perf_counter()
    query_vec = embed_query(row["question"])
    stages["embed"] = time.perf_counter() - t0

    gold = [data.title_ids[t] for t in row["gold_titles"]]
    assert k >= len(gold), f"oracle needs k >= {len(gold)} (the gold passages), got k={k}"
    t0 = time.perf_counter()
    found = retrieve(index, query_vec, k + len(gold))
    stages["search"] = time.perf_counter() - t0

    ids = gold + [i for i in found if i not in gold][: k - len(gold)]
    ids = [ids[j] for j in np.random.default_rng(row["qid"]).permutation(len(ids))]
    passages = [(data.titles[i], data.texts[i]) for i in ids]

    t0 = time.perf_counter()
    text, stats = llm.generate(build_prompt(row["question"], passages))
    stages["generate"] = time.perf_counter() - t0

    return {"answer": text, "ids": ids, "stages": stages, "llm_calls": [stats]}


# Official HotpotQA answer metrics, plus retrieval recall over gold titles.


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction, gold):
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1(prediction, gold):
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    special = ("yes", "no", "noanswer")
    if (pred in special or ref in special) and pred != ref:
        return 0.0
    pred_toks, ref_toks = pred.split(), ref.split()
    same = sum(
        (collections.Counter(pred_toks) & collections.Counter(ref_toks)).values()
    )
    if same == 0:
        return 0.0
    p, r = same / len(pred_toks), same / len(ref_toks)
    return 2 * p * r / (p + r)


def recall(retrieved_titles, gold_titles):
    """Fraction of the gold titles that appear among the retrieved titles."""
    gold = set(gold_titles)
    return len(gold & set(retrieved_titles)) / len(gold)


# Results files: one JSON per run, {"config": ..., "records": [one per question]}.


def score(row, result, wall_s, data):
    """
    Build the results-file record for one answered question.

    Parameters:
    row: the question dict (from RAGData.qa)
    result: the dict returned by answer(), multihop() or oracle()
    wall_s: end-to-end time for the question in seconds
    data: the RAGData

    Returns:
    A dict with the question, the result, and its em, f1 and recall
    """
    record = {
        "qid": row["qid"],
        "type": row["type"],
        "question": row["question"],
        "gold": row["answer"],
        "answer": result["answer"],
        "ids": result["ids"],
        "em": exact_match(result["answer"], row["answer"]),
        "f1": f1(result["answer"], row["answer"]),
        "recall": recall([data.titles[i] for i in result["ids"]], row["gold_titles"]),
        "wall_s": wall_s,
        "stages": result["stages"],
        "llm_calls": result["llm_calls"],
    }
    if "lookup" in result:
        record["lookup"] = result["lookup"]
    return record


def save_run(path, config, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        # indent=1 so the file can be read in an editor, one field per line
        json.dump({"config": config, "records": records}, f, indent=1)


def load_run(path):
    """Returns (config, records) for a results file written by save_run."""
    with open(path) as f:
        run = json.load(f)
    return run["config"], run["records"]


def summarize(records):
    """Mean em, f1, recall, wall time, per-stage time and LLM stats over records."""
    def mean(values):
        return float(np.mean(list(values)))

    summary = {m: mean(r[m] for r in records) for m in ("em", "f1", "recall", "wall_s")}
    for stage in records[0]["stages"]:
        summary[stage + "_s"] = mean(r["stages"][stage] for r in records)
    for stat in ("prompt_tokens", "prefill_s", "decode_s", "server_s"):
        summary["llm_" + stat] = mean(sum(c[stat] for c in r["llm_calls"]) for r in records)
    return summary
