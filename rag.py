import argparse
import functools
import glob
import os
import time

from tqdm import tqdm

from ann_index import MyIndexFlat, MyIndexIVF, MyIndexPQ
from rag_pipeline import answer, multihop
from rag_plots import RAGPlotHelper
from rag_utils import RAGData, embed_query, make_llm, oracle, save_run, score, summarize

# defaults
k = 5  # passages retrieved and given to the LLM (top-k)
num_questions = 300

# Experiments used for generating plots. Index None means selected on the CLI
EXPERIMENTS = {
    "indexes": (
        [("flat", "topk", 5), ("ivf", "topk", 5), ("pq", "topk", 5)],
        ["amdahl"],
    ),
    "k": (
        [(None, "topk", sweep_k) for sweep_k in (1, 3, 5, 7, 10)],
        ["llm", "k"],
    ),
    "multihop": (
        [(None, "topk", 5), (None, "topk", 7), (None, "multihop", 5), (None, "oracle", 5)],
        ["tradeoff", "hops", "types"],
    ),
}
PLOTS = ["amdahl", "llm", "k", "tradeoff", "hops", "types"]


def parse_arguments():
    # Index params, shared by every command that builds an index
    index_params = argparse.ArgumentParser(add_help=False)
    index_params.add_argument("--ncluster", type=int, default=128, help="IVF clusters")
    index_params.add_argument("--nprobe", type=int, default=8, help="IVF clusters searched")
    index_params.add_argument("--m", type=int, default=48, help="PQ subvectors")
    index_params.add_argument("--nbits", type=int, default=8, help="PQ bits per code")

    parser = argparse.ArgumentParser(
        usage=(
            'rag.py query --index {flat,ivf,pq} --question "..." [--k K] [--strategy S]\n'
            "       rag.py run --index {flat,ivf,pq} --out DIR [--k K] [--strategy S] [--n N]\n"
            "       rag.py experiment {indexes,k,multihop} --out DIR [--index I] [--n N]\n"
            "       rag.py plot --analysis {amdahl,llm,k,tradeoff,hops,types} PATH... [--out DIR]\n"
            "query, run and experiment also take --ncluster, --nprobe, --m, --nbits"
        )
    )
    # prog: otherwise subcommand errors repeat the whole usage text above
    subparsers = parser.add_subparsers(dest="command", required=True, prog="rag.py")

    query_parser = subparsers.add_parser(
        "query", parents=[index_params], help="Answer one question and show how"
    )
    query_parser.add_argument(
        "--index", required=True, choices=["flat", "ivf", "pq"], help="Index to use"
    )
    query_parser.add_argument("--question", required=True, help="Question to answer")
    query_parser.add_argument(
        "--k", type=int, default=k, help="Passages retrieved and given to the LLM (top-k)"
    )
    query_parser.add_argument(
        "--strategy", default="topk", choices=["topk", "multihop"], help="RAG strategy"
    )

    run_parser = subparsers.add_parser(
        "run", parents=[index_params], help="Evaluate one configuration over N questions"
    )
    run_parser.add_argument(
        "--index", required=True, choices=["flat", "ivf", "pq"], help="Index to use"
    )
    run_parser.add_argument("--out", required=True, help="Directory for the results file")
    run_parser.add_argument(
        "--k", type=int, default=k, help="Passages retrieved and given to the LLM (top-k)"
    )
    run_parser.add_argument(
        "--strategy",
        default="topk",
        choices=["topk", "multihop", "oracle"],
        help="RAG strategy (oracle: both gold passages always in the context)",
    )
    run_parser.add_argument(
        "--n", type=int, default=num_questions, help="Number of questions"
    )

    experiment_parser = subparsers.add_parser(
        "experiment",
        parents=[index_params],
        help="Run one of the assignment's experiments and make its plots",
    )
    experiment_parser.add_argument("name", choices=list(EXPERIMENTS), help="Experiment")
    experiment_parser.add_argument(
        "--out", required=True, help="Directory for the results files and plots"
    )
    experiment_parser.add_argument(
        "--index",
        default="flat",
        choices=["flat", "ivf", "pq"],
        help="Index for the k and multihop experiments (indexes uses all three)",
    )
    experiment_parser.add_argument(
        "--n", type=int, default=num_questions, help="Number of questions"
    )

    plot_parser = subparsers.add_parser(
        "plot", help="Plot any results files (experiments make their own plots)"
    )
    plot_parser.add_argument("--analysis", required=True, choices=PLOTS, help="Plot to make")
    plot_parser.add_argument(
        "paths", nargs="+", help="Results files, or directories of them; one bar/point each"
    )
    plot_parser.add_argument("--out", default="plots", help="Directory to save the plot in")

    return parser.parse_args()


def index_tag(index_name, params):
    """The index name with its params, e.g. ivf-nc128-np8."""
    if index_name == "ivf":
        return f"ivf-nc{params.ncluster}-np{params.nprobe}"
    if index_name == "pq":
        return f"pq-m{params.m}-b{params.nbits}"
    return index_name


def build_index(index_name, params, embeddings):
    """params: the parsed command line (uses ncluster, nprobe, m, nbits)."""
    print(f"Building {index_tag(index_name, params)} over {embeddings.shape[0]} passages...")
    if index_name == "flat":
        index = MyIndexFlat()
        index.train(embeddings)
    elif index_name == "ivf":
        index = MyIndexIVF()
        index.train(embeddings, params.ncluster)
        # retrieve() calls index.search(query, k), so bind nprobe here
        index.search = functools.partial(index.search, nprobe=params.nprobe)
    elif index_name == "pq":
        index = MyIndexPQ(params.m)
        index.train(embeddings, params.nbits)
    else:
        raise ValueError(f"unknown index: {index_name!r}")
    return index


def start_llm():
    """The LLM, with it and the embedding model loaded so no timed call pays for that."""
    llm = make_llm()
    embed_query("warmup")
    llm.warmup()
    return llm


def run_strategy(strategy, row, index, k, data, llm):
    """
    Answer one question with a strategy and return the result dict.

    Parameters:
    strategy: "topk", "multihop" or "oracle"
    row: the question dict; row["question"] is the question string (for `query`, that's
         the only key)
    index: your trained Part 1 index over data.embeddings
    k: the --k from the command line (passages given to the LLM)
    data: the RAGData holding the corpus (data.titles, data.texts)
    llm: an OllamaClient (or FakeLLM), already warmed up
    """
    if strategy == "multihop":
        # TODO-[Part 5.3]: call your multihop() from rag_pipeline.py
        raise NotImplementedError("multihop")
    if strategy == "oracle":
        return oracle(row, index, k, data, llm)
    # TODO-[Part 5.1]: call your answer() from rag_pipeline.py
    raise NotImplementedError("answer")


def run_query(index_name, params, question, k, strategy, data):
    index = build_index(index_name, params, data.embeddings)
    llm = start_llm()

    result = run_strategy(strategy, {"question": question}, index, k, data, llm)
    print(f"\nQuestion: {question}")
    if "lookup" in result:
        print(f"Second-hop lookup: {result['lookup']}")
    print("Retrieved:")
    for i in result["ids"]:
        print(f"  [{data.titles[i]}] {data.texts[i][:100]}...")
    print(f"Answer: {result['answer']}")
    print(
        "Stage times: "
        + ", ".join(f"{s} {t:.3f}s" for s, t in result["stages"].items())
    )
    for stats in result["llm_calls"]:
        print(
            f"LLM ({llm.model}): {stats['prompt_tokens']} prompt tokens, "
            f"{stats['output_tokens']} output tokens, prefill {stats['prefill_s']:.3f}s, "
            f"decode {stats['decode_s']:.3f}s"
        )


def run_config(tag, index, strategy, k, n, out, data, llm):
    """Answer n questions with one configuration, save the results in out, return the path."""
    config = {
        "index": tag,
        "strategy": strategy,
        "k": k,
        "model": llm.model,
        "gpu": os.environ.get("RAG_GPU") or "unknown",  # set by the job script
    }
    records = []
    for row in tqdm(data.sample_questions(n), desc=f"{tag} {strategy} k={k}"):
        t0 = time.perf_counter()
        result = run_strategy(strategy, row, index, k, data, llm)
        wall_s = time.perf_counter() - t0
        records.append(score(row, result, wall_s, data))

    path = os.path.join(out, f"{tag}_{strategy}_k{k}.json")
    save_run(path, config, records)
    print(f"\n{path}, {llm.model}, {n} questions (mean per question):")
    for name, value in summarize(records).items():
        print(f"  {name:18s} {value:.3f}")
    return path


def run_experiment(name, chosen_index, params, n, out, data):
    runs, plots = EXPERIMENTS[name]
    llm = start_llm()
    indexes = {}  # each index is built once and shared by the runs that use it
    paths = []
    for i, (index_name, strategy, run_k) in enumerate(runs):
        if i > 0:
            # Each run asks the same questions, so the server's prompt cache would make this
            # run's prefill look faster than the previous one's. Start it from a cold cache.
            llm.unload()
            llm.warmup()
        index_name = index_name or chosen_index
        if index_name not in indexes:
            indexes[index_name] = build_index(index_name, params, data.embeddings)
        tag = index_tag(index_name, params)
        paths.append(run_config(tag, indexes[index_name], strategy, run_k, n, out, data, llm))
    for analysis in plots:
        plot(analysis, paths, out)


def plot(analysis, paths, out):
    """Make one plot from results files (or directories of them) and save it in out."""
    files = []
    for path in paths:
        files += sorted(glob.glob(os.path.join(path, "*.json"))) if os.path.isdir(path) else [path]
    if not files:
        raise FileNotFoundError(f"no results files in {paths}")
    plot_helper = RAGPlotHelper(files, out)
    {
        "amdahl": plot_helper.generate_amdahl_plot,
        "llm": plot_helper.generate_llm_plot,
        "k": plot_helper.generate_k_plot,
        "tradeoff": plot_helper.generate_tradeoff_plot,
        "hops": plot_helper.generate_hops_plot,
        "types": plot_helper.generate_types_plot,
    }[analysis]()


if __name__ == "__main__":
    args = parse_arguments()

    if args.command == "plot":
        plot(args.analysis, args.paths, args.out)
    elif args.command == "query":
        run_query(args.index, args, args.question, args.k, args.strategy, RAGData())
    elif args.command == "run":
        data = RAGData()
        index = build_index(args.index, args, data.embeddings)
        tag = index_tag(args.index, args)
        run_config(tag, index, args.strategy, args.k, args.n, args.out, data, start_llm())
    elif args.command == "experiment":
        run_experiment(args.name, args.index, args, args.n, args.out, RAGData())
