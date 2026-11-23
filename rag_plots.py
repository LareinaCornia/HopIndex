import os

import matplotlib.pyplot as plt
import numpy as np

from rag_utils import load_run, summarize

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc9", "#d4a017"]
GRID_COLOR = "#dcdcd6"
TEXT_COLOR = "#52514e"
# Legend names for the timed stages (the keys students fill in stay embed/search/bridge/generate)
STAGE_LABELS = {"embed": "embed", "search": "search", "bridge": "LLM: bridge call",
                "generate": "LLM: answer call"}


class RAGPlotHelper:
    """
    Plots for the Part A and Part B analyses. Reads saved results files only; never calls
    the LLM or builds an index.
    """

    def __init__(self, paths, out):
        """
        paths: results files, each one run over the same questions
        out: directory to save the plots in
        """
        self.out = out
        self.runs = [(path, *load_run(path)) for path in paths]
        if len({frozenset(r["qid"] for r in records) for _, _, records in self.runs}) > 1:
            raise ValueError("these runs answered different questions; use the same --n")
        # Plain top-k first, then multi-hop, then the oracle (stable, so an explicit order
        # or the alphabetical order of a directory is kept within each group)
        rank = {"topk": 0, "multihop": 1, "oracle": 2}
        self.runs.sort(key=lambda run: rank.get(run[1]["strategy"], 3))
        self.models = ", ".join(sorted({config["model"] for _, config, _ in self.runs}))
        self.n = len(self.runs[0][2])
        # Latencies are only comparable between runs timed on the same GPU type
        self.gpus = sorted({config.get("gpu") or "unknown GPU" for _, config, _ in self.runs})
        if len(self.gpus) > 1:
            print(f"WARNING: these runs were timed on different GPUs ({', '.join(self.gpus)}); "
                  "their latencies are not comparable")

    def _labels(self, label_fn):
        """One label per run; runs whose labels clash are told apart by their directory."""
        labels = [label_fn(config) for _, config, _ in self.runs]
        return [
            f"{label} ({os.path.basename(os.path.dirname(path))})"
            if labels.count(label) > 1 else label
            for label, (path, _, _) in zip(labels, self.runs)
        ]

    def _summaries(self):
        return [summarize(records) for _, _, records in self.runs]

    def _sort_by_k(self):
        self.runs.sort(key=lambda run: run[1]["k"])

    @staticmethod
    def _strategy_label(config):
        if config["strategy"] == "topk":
            return f"top-{config['k']}"
        return f"{config['strategy']} (k={config['k']})"

    def _title(self, text):
        text += "\n" + ", ".join(self.gpus)
        if len(self.gpus) > 1:
            text += " [MIXED GPUs: latencies not comparable]"
        if "fake" in self.models:
            text += "\n[FAKE LLM: not real measurements]"
        return text

    def _style_axes(self, ax, x_label, y_label):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COLOR)

    def _stacked_bars(self, ax, labels, segments, colors=COLORS):
        """segments: {segment name: list of values, one per bar}, drawn bottom-up."""
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        for color, (name, values) in zip(colors, segments.items()):
            ax.bar(x, values, bottom=bottom, width=0.6, color=color, label=name,
                   edgecolor="white", linewidth=2)
            bottom += np.array(values)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, bottom.max() * 1.25)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=len(segments))
        return x, bottom

    def _save(self, fig, filename):
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        os.makedirs(self.out, exist_ok=True)
        path = os.path.join(self.out, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Plot saved as {path}")

    def generate_amdahl_plot(self):
        """Mean per-stage latency for each run (one per index): how much of the total is search?"""
        summaries = self._summaries()
        stages = ["embed", "search", "generate"]
        segments = {STAGE_LABELS[stage]: [s[stage + "_s"] for s in summaries] for stage in stages}

        fig, ax = plt.subplots(figsize=(7, 5))
        x, totals = self._stacked_bars(ax, self._labels(lambda c: c["index"]), segments)
        for xi, total, search in zip(x, totals, segments["search"]):
            ax.text(float(xi), total, f"{total:.2f}s\nsearch {100 * search / total:.0f}%",
                    ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)
        self._style_axes(ax, "index", "mean latency per question (s)")
        ax.set_title(self._title(f"Where RAG time goes ({self.models}, {self.n} questions)"))
        self._save(fig, "rag_amdahl.png")

    def generate_llm_plot(self):
        """Mean LLM call time split into prefill / decode / everything else, per k."""
        self._sort_by_k()
        summaries = self._summaries()
        prefill = [s["llm_prefill_s"] for s in summaries]
        decode = [s["llm_decode_s"] for s in summaries]
        overhead = [s["generate_s"] - p - d for s, p, d in zip(summaries, prefill, decode)]

        fig, ax = plt.subplots(figsize=(7, 5))
        segments = {"prefill": prefill, "decode": decode, "overhead": overhead}
        x, totals = self._stacked_bars(ax, self._labels(lambda c: str(c["k"])), segments)
        for xi, total, s in zip(x, totals, summaries):
            ax.text(float(xi), total, f"{s['llm_prompt_tokens']:.0f} tok", ha="center",
                    va="bottom", fontsize=9, color=TEXT_COLOR)
        self._style_axes(ax, "k (passages given to the LLM)", "mean LLM call time (s)")
        ax.set_title(self._title(f"Inside the LLM call ({self.models}, {self.n} questions)"))
        self._save(fig, "rag_llm_time.png")

    def generate_k_plot(self):
        """Answer quality and latency as k grows (one run per k)."""
        self._sort_by_k()
        k_list = [config["k"] for _, config, _ in self.runs]
        summaries = self._summaries()

        fig, (ax_quality, ax_latency) = plt.subplots(1, 2, figsize=(11, 4.5))
        for color, (metric, label) in zip(COLORS, [("recall", "retrieval recall"),
                                                   ("em", "EM"), ("f1", "F1")]):
            ax_quality.plot(k_list, [s[metric] for s in summaries], marker="o",
                            linewidth=2, markersize=6, color=color, label=label)
        ax_quality.set_ylim(0, 1)
        ax_quality.legend(frameon=False, loc="lower right")
        self._style_axes(ax_quality, "k (passages given to the LLM)", "score")
        ax_quality.set_title("Quality vs k")

        ax_latency.plot(k_list, [s["wall_s"] for s in summaries], marker="o",
                        linewidth=2, markersize=6, color=COLORS[0])
        ax_latency.set_ylim(0, max(s["wall_s"] for s in summaries) * 1.2)
        self._style_axes(ax_latency, "k (passages given to the LLM)",
                         "mean latency per question (s)")
        ax_latency.set_title("Latency vs k")
        for ax in (ax_quality, ax_latency):
            ax.set_xticks(k_list)
        fig.suptitle(self._title(f"{self.models}, {self.n} questions"))
        self._save(fig, "rag_k_sweep.png")

    def generate_tradeoff_plot(self):
        """
        Recall and EM against latency for each run (the Part B configurations). The oracle's
        latency is an artifact of how it is built, so it is drawn as a ceiling line instead.
        """
        summaries = self._summaries()
        labels = self._labels(self._strategy_label)
        is_oracle = [config["strategy"] == "oracle" for _, config, _ in self.runs]
        points = [(l, s) for l, s, o in zip(labels, summaries, is_oracle) if not o]
        oracles = [(l, s) for l, s, o in zip(labels, summaries, is_oracle) if o]

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
        for ax, (metric, label) in zip(axes, [("recall", "retrieval recall"), ("em", "EM")]):
            for color, (run_label, s) in zip(COLORS, points):
                ax.scatter(s["wall_s"], s[metric], s=60, color=color, zorder=3, label=run_label)
            for color, (run_label, s) in zip(COLORS[len(points):], oracles):
                ax.axhline(s[metric], color=color, linestyle="--", linewidth=1.5, zorder=2,
                           label=f"{run_label}: perfect retrieval")
            ax.set_xlim(0, max(s["wall_s"] for _, s in points) * 1.3)
            self._style_axes(ax, "mean latency per question (s)", label)
            ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
            ax.set_title(f"{label} vs latency")
        axes[0].set_ylim(0, 1.05)
        ems = [s["em"] for s in summaries]
        axes[1].set_ylim(max(0, min(ems) - 0.1), min(1, max(ems) + 0.1))
        fig.legend(*axes[0].get_legend_handles_labels(), frameon=False,
                   loc="outside lower center", ncol=len(labels))
        fig.suptitle(self._title(f"Part B trade-off ({self.models}, {self.n} questions)"))
        self._save(fig, "rag_tradeoff.png")

    def generate_hops_plot(self):
        """
        Mean per-stage latency of each run (top-k vs multi-hop): what does a hop cost? The
        oracle is left out: its latency is an artifact of how it is built.
        """
        keep = [i for i, (_, config, _) in enumerate(self.runs) if config["strategy"] != "oracle"]
        summaries = [self._summaries()[i] for i in keep]
        labels = [self._labels(self._strategy_label)[i] for i in keep]
        stages = ["embed", "search", "bridge", "generate"]
        segments = {STAGE_LABELS[stage]: [s.get(stage + "_s", 0.0) for s in summaries]
                    for stage in stages}

        fig, ax = plt.subplots(figsize=(7, 5))
        # Same colors as the Amdahl plot for the shared stages
        colors = [COLORS[0], COLORS[1], COLORS[3], COLORS[2]]
        x, totals = self._stacked_bars(ax, labels, segments, colors)
        for xi, total in zip(x, totals):
            ax.text(float(xi), total, f"{total:.2f}s", ha="center", va="bottom",
                    fontsize=9, color=TEXT_COLOR)
        self._style_axes(ax, "strategy", "mean latency per question (s)")
        ax.set_title(self._title(f"Cost of a second hop ({self.models}, {self.n} questions)"))
        self._save(fig, "rag_hops.png")

    def generate_types_plot(self):
        """EM by HotpotQA question type (bridge vs comparison) for each run."""
        types = ["bridge", "comparison"]
        em = {}
        for config_label, (_, _, records) in zip(self._labels(self._strategy_label), self.runs):
            em[config_label] = [np.mean([r["em"] for r in records if r["type"] == t]) for t in types]
        counts = [sum(r["type"] == t for r in records) for t in types]

        fig, ax = plt.subplots(figsize=(8, 5))
        width = 0.8 / len(self.runs)
        for i, (color, config_label) in enumerate(zip(COLORS, em)):
            ax.bar(np.arange(len(types)) + (i - (len(self.runs) - 1) / 2) * width,
                   em[config_label], width=width, color=color, label=config_label,
                   edgecolor="white", linewidth=1)
        ax.set_xticks(np.arange(len(types)))
        ax.set_xticklabels([f"{t} ({c} questions)" for t, c in zip(types, counts)])
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=len(self.runs))
        self._style_axes(ax, "question type", "EM")
        ax.set_title(self._title(f"Where the gain lands ({self.models}, {self.n} questions)"))
        self._save(fig, "rag_types.png")
