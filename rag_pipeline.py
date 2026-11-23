import time

from rag_utils import build_prompt, embed_query, retrieve


def answer():
    """
    Answer a question with RAG: embed it, retrieve passages with your Part 1 index, build
    a prompt with build_prompt(), and ask the LLM. Time each stage. Choose the parameters
    yourself, and call this from run_strategy() in rag.py.

    Returns:
    A dict with:
        - answer: the LLM's answer string
        - ids: the retrieved corpus ids, in the order given to the LLM
        - stages: wall time in seconds of each stage: {"embed", "search", "generate"}
        - llm_calls: a list with the stats dict returned by llm.generate()
    """
    stages = {"embed": 0.0, "search": 0.0, "generate": 0.0}

    # TODO-[Part 5.1]: your code here
    raise NotImplementedError("answer")


# Given: the bridge prompt and lookup parser behind our reference numbers. Use them as they
# are (recommended), or improve on them if you want to.


def bridge_prompt(question, passages):
    """
    Build the prompt that asks the LLM which Wikipedia article to look up next.

    Parameters:
    question: the question string
    passages: a list of (title, text) pairs retrieved so far

    Returns:
    The prompt string
    """
    context = "\n\n".join(f"[{title}]\n{text}" for title, text in passages)
    return (
        "You are searching Wikipedia to answer a question. Below are the articles found "
        "so far. They may mention an entity whose OWN article is needed to answer the "
        "question but has not been retrieved yet.\n\n"
        f"Articles so far:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Which Wikipedia article should be looked up next? Reply with ONLY the article "
        "title (a name, place, work, or organization), nothing else.\n"
        "Next article title:"
    )


def parse_lookup(reply, question):
    """
    Turn the LLM's reply to bridge_prompt into a search query: its first line without
    quotes or brackets, or the question itself if the reply is empty.
    """
    lookup = reply.splitlines()[0].strip(" \"'[]*.") if reply else ""
    return lookup or question


def multihop():
    """
    Answer a question with more than one retrieval hop: retrieve, have the LLM name what
    to look up next (bridge_prompt() and parse_lookup() above do this part), retrieve
    that, and answer from everything collected. The rest of the design is yours (see the
    handout). Choose the parameters yourself, and call this from run_strategy() in rag.py.

    Returns:
    A dict with:
        - answer: the LLM's answer string
        - ids: the corpus ids given to the LLM for the answer, in prompt order
        - stages: wall time in seconds of each stage, summed over hops:
          {"embed", "search", "bridge" (every LLM call before the answer), "generate"}
        - llm_calls: the stats dicts of every llm.generate() call, in order
        - lookup (optional): the second hop's search query, shown by `rag.py query`
    """
    stages = {"embed": 0.0, "search": 0.0, "bridge": 0.0, "generate": 0.0}

    # TODO-[Part 5.3]: your code here
    raise NotImplementedError("multihop")
