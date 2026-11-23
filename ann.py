import argparse
import faiss
import numpy as np

from ann_index import MyIndexFlat, MyIndexIVF, MyIndexPQ, save_index, load_index
from utils import FileHelper as FileHelper
from utils import Evaluation as Evaluation
from utils import PlotHelper as PlotHelper

# global variables
ncluster = 128
nprobe = 8
m = 4
k = 10
nbits = 8

memory_bound = 500000

score_map = {
    "flat": 100,
    "ivf": 95,
    "pq": 75
}

def parse_arguments():
    parser = argparse.ArgumentParser(
    usage=(
            "ann.py train --index {flat,ivf,pq}\n"
            "       ann.py check --index {flat,ivf,pq,insert}\n"
            "       ann.py metric --index {flat,ivf,pq} --metric {accuracy,memory,speed}\n"
            "       ann.py plot --index {flat,ivf,pq,insert} --metric {accuracy,memory,speed}"
            "       ann.py faiss"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train index and save results to file")
    train_parser.add_argument("--index", required=True, choices=["flat", "ivf", "pq"], help="Index to train")

    check_parser = subparsers.add_parser("check", help="Check correctness of the index")
    check_parser.add_argument("--index", required=True, choices=["flat", "ivf", "pq", "insert"], help="Index to check")

    metric_parser = subparsers.add_parser("metric", help="Output specific metric of the index")
    metric_parser.add_argument("--index", required=True, choices=["flat", "ivf", "pq"], help="Index type")
    metric_parser.add_argument("--metric", required=True, choices=["speed", "memory", "accuracy"], help="Metric to output")

    plot_parser = subparsers.add_parser("plot", help="Plot graphs")
    plot_parser.add_argument("--index", required=True, choices=["flat", "ivf", "pq", "insert"], help="Index type")
    plot_parser.add_argument("--metric", required=True, choices=["speed", "memory", "accuracy", "all"], help="Metric to plot")

    faiss_parser = subparsers.add_parser("faiss", help="Get faiss query time")
    
    args = parser.parse_args()
    return args

def get_faiss_query_time(base_vecs, query_vecs):
    evaluation = Evaluation(base_vecs, query_vecs)
    print("Getting faiss query time...")
    index = faiss.IndexFlatL2(base_vecs.shape[1])
    index.add(base_vecs)
    print(f"Faiss flat index query time: {evaluation.get_average_query_time(index)}")

    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(base_vecs.shape[1]), base_vecs.shape[1], ncluster, faiss.METRIC_L2)
    index.train(base_vecs)
    index.add(base_vecs)
    index.nprobe = nprobe
    print(f"Faiss ivf index query time: {evaluation.get_average_query_time(index)}")

    index = faiss.IndexPQ(base_vecs.shape[1], m, nbits)
    index.train(base_vecs)
    index.add(base_vecs)
    print(f"Faiss pq index query time: {evaluation.get_average_query_time(index)}")

def check_insertion(base_vecs, query_vecs):
    print("Checking flat index insertion...")

    if not check_index("flat", base_vecs, query_vecs, insert=True):
        print("Flat index insertion failed!")
        return

    print("Checking ivf index insertion...")
    if not check_index("ivf", base_vecs, query_vecs, insert=True):
        print("IVF index insertion failed!")
        return

    print("Checking pq index insertion...")
    if not check_index("pq", base_vecs, query_vecs, insert=True):
        print("PQ index insertion failed!")
        return
    
    print("Insertion check passed for all indexes!")

def train_index(index_name, base):
    print(f"Training {index_name} index...")
    if index_name == "flat":
        index = MyIndexFlat()
        index.train(base)
        save_index(index, "saved/flat_index.joblib")
        
    elif index_name == "ivf":
        index = MyIndexIVF()
        index.train(base, nclusters=ncluster)
        save_index(index, "saved/ivf_index.joblib")

    elif index_name == "pq":
        index = MyIndexPQ(m=m)
        index.train(base, nbits=nbits)
        save_index(index, "saved/pq_index.joblib")
    
    print(f"Trained {index_name} index saved.")

def check_index(index_name, base, querys, insert = False):
    print(f"Checking correctness of {index_name}...")
    score = 0
    index, refered_index = None, None
    standard_index = faiss.IndexFlatL2(base.shape[1])
    standard_index.add(base)

    fileHelper = FileHelper()
    insert_vecs = fileHelper.read_fvecs('./siftsmall/siftsmall_learn.fvecs')
    insert_vecs = insert_vecs[:1000]

    if index_name == "flat":
        index = load_index("saved/flat_index.joblib")
        if index == None:
            index = MyIndexFlat()
            index.train(base)
        if insert:
            index.insert(insert_vecs)

    elif index_name == "ivf":
        index = load_index("saved/ivf_index.joblib")
        if index == None:
            index = MyIndexIVF()
            index.train(base, nclusters=ncluster)
        refered_index = faiss.IndexIVFFlat(faiss.IndexFlatL2(base.shape[1]), base.shape[1], ncluster, faiss.METRIC_L2)
        refered_index.train(base)
        refered_index.add(base)
        refered_index.nprobe = 8
        if insert:
            index.insert(insert_vecs)

    elif index_name == "pq":
        index = load_index("saved/pq_index.joblib")
        if index == None:
            index = MyIndexPQ(m=m)
            index.train(base, nbits=nbits)
        
        # check memory usage
        memory = Evaluation(base, querys).get_memory(index)
        if memory > memory_bound:
            print(f"Memory usage of {index_name} index is too high: {memory}.")
            return 0

        refered_index = faiss.IndexPQ(base.shape[1], 8, 8)
        refered_index.train(base)
        refered_index.add(base)
        if insert:
            index.insert(insert_vecs)

    elif index_name == "insert":
        return check_insertion(base, querys)

    for i, query in enumerate(querys):
        if index_name == "ivf":
            D, I = index.search(query, 10, nprobe=8)
        else:
            D, I = index.search(query, 10)
        result = [{"index": int(I[i]), "distance": round(float(D[i]), 2)} for i in range(len(I))]
        query = query.reshape(1, query.shape[0])
        standard_D, standard_I = standard_index.search(query, 1)
        if standard_I[0] in I:
            # check pass
            score += 1
        else:
            if index_name == "flat":
                print(f"Point failed to match for query: query[{i}]")
                print(f"Nearest point: index - {int(standard_I[0])}, distance - {round(float(standard_D[0]), 2)}")
                print(f"Your result: {result}")
                print("=================================================")
            refered_D, refered_I = refered_index.search(query, 1)
            if refered_I[0] in I:
                score += 1
            else:
                print(f"Point failed to match for query: query[{i}]")
                print(f"Nearest point: index - {int(standard_I[0])}, distance - {round(float(standard_D[0]), 2)}")
                print(f"Referred point: index - {int(refered_I[0])}, distance - {round(float(refered_D[0]), 2)}")
                print(f"Your result: {result}")
                print("=================================================")

    if score >= score_map[index_name]:
        print(f"Test passed for {index_name} index!")
        print(f"Score: {score}")
        return 1
    else:
        print(f"Test failed for {index_name} index!")
        print(f"Score: {score}")
        return 0

def evaluate_metric(index_name, metric, base, query):
    print(f"Evaluating {metric} of {index_name}...")
    index = None
    res = None
    eval_helper = Evaluation(base, query)
    # A saved index is reused for speed and accuracy, but memory always uses a freshly
    # trained one: an index reloaded from disk measures about twice its real size.
    load = load_index if metric != "memory" else (lambda path: None)
    
    if index_name == "flat":
        index = load("saved/flat_index.joblib")
        if index == None:
            index = MyIndexFlat()
            index.train(base)
    elif index_name == "ivf":
        index = load("saved/ivf_index.joblib")
        if index == None:
            index = MyIndexIVF()
            index.train(base, nclusters=ncluster)
    elif index_name == "pq":
        index = load("saved/pq_index.joblib")
        if index == None:
            index = MyIndexPQ(m=m)
            index.train(base, nbits=nbits)

    if metric == "speed":
        if index_name == "ivf":
            res = eval_helper.get_average_query_time(index, nprobe=8)
        else:
            res = eval_helper.get_average_query_time(index)
        print(f"The average search speed of {index_name} index is {res}.")
    if metric == "memory":
        res = eval_helper.get_memory(index)
        print(f"The memory usage of {index_name} index is {res}.")
    if metric == "accuracy":
        if index_name == "ivf":
            res = eval_helper.get_accuracy(index, nprobe=8)
        else:
            res = eval_helper.get_accuracy(index)
        print(f"The accuracy of {index_name} index is {res}.")

def plot_metric(index_name, metric, base, query):
    print(f"Plotting {metric} for {index_name}, this may take a while...")
    eval_helper = Evaluation(base, query)
    plot_helper = PlotHelper(eval_helper)

    if index_name == 'flat':
        if metric == "speed":
            plot_helper.generate_sz_time_plot(MyIndexFlat)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Supported metrics is speed.")
    elif index_name == 'ivf':
        if metric == "speed":
            plot_helper.generate_nprobe_ncluster_time_plot(MyIndexIVF)
        elif metric == "accuracy":
            plot_helper.generate_nprobe_ncluster_accuracy_plot(MyIndexIVF)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Supported metrics are speed and accuracy.")
    elif index_name == 'pq':
        if metric == "speed":
            plot_helper.generate_m_nbit_time_plot(MyIndexPQ)
        elif metric == "accuracy":
            plot_helper.generate_m_nbit_accuracy_plot(MyIndexPQ)
        elif metric == "memory":
            plot_helper.generate_m_memory_plot(MyIndexPQ)
        elif metric == "all":
            plot_helper.generate_all_pq(MyIndexPQ)
    elif index_name == 'insert':
        if metric == "accuracy":
            plot_helper.generate_insert_accuracy_plot(MyIndexIVF)
        elif metric == "speed":
            indexs = [MyIndexFlat, MyIndexIVF, MyIndexPQ]
            plot_helper.generate_insert_time_plot(indexs)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Supported metrics is accuracy.")

if __name__ == "__main__":
    # download dataset if need
    filehelper = FileHelper()
    if not filehelper.is_file_exists("./siftsmall.tar.gz"):
        filehelper.download_file('ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz', "./siftsmall.tar.gz")
    if not filehelper.is_file_exists("./siftsmall/siftsmall_base.fvecs"):
        filehelper.unzip_file("./siftsmall.tar.gz")
    
    # read data into memory
    base_vecs = filehelper.read_fvecs('./siftsmall/siftsmall_base.fvecs')
    query_vecs = filehelper.read_fvecs('./siftsmall/siftsmall_query.fvecs')

    # parse arguments
    args = parse_arguments()
    if args.command == "train":
        train_index(args.index, base_vecs)
    elif args.command == "check":
        check_index(args.index, base_vecs, query_vecs)
    elif args.command == "metric":
        evaluate_metric(args.index, args.metric, base_vecs, query_vecs)
    elif args.command == "plot":
        plot_metric(args.index, args.metric, base_vecs, query_vecs)
    elif args.command == "faiss":
        get_faiss_query_time(base_vecs, query_vecs)