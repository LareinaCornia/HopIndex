import random
import numpy as np
import os

from urllib.request import urlopen
from tqdm import tqdm
from contextlib import closing
import tarfile

class FileHelper:
    def download_file(self, url, filepath):
        with closing(urlopen(url)) as r:
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024

            with open(filepath, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                while True:
                    data = r.read(block_size)
                    if not data:
                        break
                    f.write(data)
                    bar.update(len(data))

    def unzip_file(self, file):
        tar = tarfile.open(file, "r:gz")
        tar.extractall(filter="data")
        print(f"unzip {file} done!")

    def is_file_exists(self, file):
        return os.path.exists(file)

    def read_fvecs(self, fp):
        a = np.fromfile(fp, dtype='int32')
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')
    

import matplotlib.pyplot as plt
from pympler import asizeof
import faiss
import time

class Evaluation:
    def __init__(self, base_vecs, query_vecs):
        self.N = 10

        index = faiss.IndexFlatL2(base_vecs.shape[1])
        index.add(base_vecs)
        self.faiss_flat_index = index
        self.base = base_vecs
        self.query = query_vecs

    def _generate_sample(self, vecs, size):
            if size > vecs.shape[0]:
                raise ValueError("can't generate sample larger than itself!")
            sampled_rows = np.random.choice(vecs.shape[0], size=size, replace=False)
            return vecs[sampled_rows]
    
    def _compare_with_standard(self, query, indice):
        """Compare results with standard Faiss flat index, which is 100% accurate. """
        query = query.reshape(1, query.shape[0])
        _, I = self.faiss_flat_index.search(query, 1)
        if I[0] in indice:
            return True
    
    def _get_average_insert_time(self, index, new_vecs):
        round_time = 0
        for i in range(self.N):
            start_time = time.time()
            index.insert(new_vecs[1000*i:1000*(i+1)])
            end_time = time.time()
            round_time = round_time + (end_time - start_time)
        return round_time / self.N
    
    def get_average_query_time(self, index, *args, **kwargs):
        round_time = 0
        for _ in range(self.N):
            start_time = time.time()
            query = random.choice(self.query)
            if not index.__class__.__name__.startswith("My"):       # we are using faiss, need to reshape the vector
                query = query.reshape(1, query.shape[0])
            index.search(query, 1, *args, **kwargs)
            end_time = time.time()
            round_time = round_time + (end_time - start_time)
        return round(round_time / self.N, 6)

    def get_memory(self, index):
        """Return the memory size of the given index in bytes."""
        return asizeof.asizeof(index)
    
    def get_accuracy(self, index, *args, **kwargs):
        correct_count = 0
        total_count = len(self.query)

        for query in self.query:
            _, I = index.search(query, 10, *args, **kwargs)
            if self._compare_with_standard(query, I):
                correct_count += 1
        return correct_count / total_count
    
class PlotHelper:
    def __init__(self, evaluation_tool):
        self.evaluation_tool = evaluation_tool

        self.sizes = np.linspace(1000, 10000, num=10)

        self.nprobe_list = [1, 4, 8, 12]
        self.ncluster_list = [16, 32, 64, 128, 256]
        self.nbits_list = [4, 5, 6, 7, 8]

        self.m_list = [4, 8, 16, 32, 64]

        self.colors = ['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974']

    def _generate_plot(self, x_data, y_datas, labels, x_label, y_label, title):
        plt.figure(figsize=(8, 5))
        for i, y_data in enumerate(y_datas):
            plt.plot(x_data, y_data, marker='o', linestyle='-', color=self.colors[i], label=f"{labels[i]}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()

        return plt
    
    def _save_plot(self, plot, filename):
        if not os.path.exists("./plots"):
            os.makedirs("./plots")
        plot.savefig("./plots/"+filename)
        print(f"Plot saved as {filename}")

    def generate_sz_time_plot(self, IndexClass):
        times = []
        
        for size in self.sizes:
            base = self.evaluation_tool._generate_sample(self.evaluation_tool.base, int(size))
            index = IndexClass()
            index.train(base)
            times.append(self.evaluation_tool.get_average_query_time(index))
        plot = self._generate_plot(self.sizes, [times], ["flat index"], 
                                   "number of vectors",
                                   "Average Search Time (s)",
                                    "Search Time vs Number of Vectors")
        plot.show()
        self._save_plot(plot, IndexClass.__name__ + "_sz_time_plot.png")

    def generate_nprobe_ncluster_time_plot(self, IndexClass):
        times = []
        labels = []
        for ncluster in self.ncluster_list:
            round_time = []
            index = IndexClass()
            index.train(self.evaluation_tool.base, ncluster)
            for nprobe in self.nprobe_list:
                round_time.append(self.evaluation_tool.get_average_query_time(index, nprobe=nprobe))
            times.append(round_time)
            labels.append(f"ncluster={ncluster}")

        plot = self._generate_plot(self.nprobe_list, times, labels, 
                                   "nprobe", 
                                   "Average Search Time (s)",
                                   "Search Time under Different nprobe and ncluster")
        plot.show()
        self._save_plot(plot, IndexClass.__name__ + "_nprobe_ncluster_time_plot.png")

    def generate_nprobe_ncluster_accuracy_plot(self, IndexClass):
        accuracy = []
        labels = []
        for ncluster in self.ncluster_list:
            round_acc = []
            index = IndexClass()
            index.train(self.evaluation_tool.base, ncluster)
            for nprobe in self.nprobe_list:
                round_acc.append(self.evaluation_tool.get_accuracy(index, nprobe=nprobe))
            accuracy.append(round_acc)
            labels.append(f"ncluster={ncluster}")

        plot = self._generate_plot(self.nprobe_list, accuracy, labels, 
                                   "nprobe", 
                                   "accuracy",
                                   "Accuracy under Different nprobe and ncluster")
        plot.show()
        self._save_plot(plot, IndexClass.__name__ + "_nprobe_ncluster_accuracy_plot.png")

    def generate_m_nbit_time_plot(self, IndexClass, index_map = None):
        times = []
        labels = []
        for nbits in self.nbits_list:
            round_time = []
            for m in self.m_list:
                if index_map is None:
                    index = IndexClass(m)
                    index.train(self.evaluation_tool.base, nbits)
                else:
                    index = index_map[m][nbits]
                round_time.append(self.evaluation_tool.get_average_query_time(index))
            times.append(round_time)
            labels.append(f"nbits={nbits}")

        plot = self._generate_plot(self.m_list, times, labels, 
                                   "m", 
                                   "Average Search Time (s)",
                                   "Search Time under Different m and mbits")
        plot.show()
        self._save_plot(plot, IndexClass.__name__ + "_m_nbit_time_plot.png")

    def generate_m_nbit_accuracy_plot(self, IndexClass, index_map = None):
        accuracy = []
        labels = []
        for nbits in self.nbits_list:
            round_acc = []
            for m in self.m_list:
                if index_map is None:
                    index = IndexClass(m)
                    index.train(self.evaluation_tool.base, nbits)
                else:
                    index = index_map[m][nbits]
                round_acc.append(self.evaluation_tool.get_accuracy(index))
            accuracy.append(round_acc)
            labels.append(f"nbits={nbits}")

        plot = self._generate_plot(self.m_list, accuracy, labels, 
                                   "m", 
                                   "accuracy",
                                   "Accuracy under Different m and mbits")
        plot.show()
        self._save_plot(plot, IndexClass.__name__ + "_m_nbit_accuracy_plot.png")

    def generate_m_memory_plot(self, IndexClass, index_map = None):
        memory = []
        for m in self.m_list:
            if index_map is None:
                index = IndexClass(m)
                index.train(self.evaluation_tool.base, 8)
            else:
                index = index_map[m][8]
            memory.append(self.evaluation_tool.get_memory(index))
        plot = self._generate_plot(self.m_list, [memory], ["pq"], 
                                   "m",
                                   "memory(bytes)",
                                    "Memory vs m")
        plot.show()
        self._save_plot(plot, IndexClass.__name__ + "_m_memory_plot.png")

    def generate_all_pq(self, IndexClass):
        index_map = {}
        for m in self.m_list:
            index_map[m] = {}
            for nbits in self.nbits_list:
                index = IndexClass(m)
                index.train(self.evaluation_tool.base, nbits)
                index_map[m][nbits] = index
        self.generate_m_nbit_time_plot(IndexClass, index_map)
        self.generate_m_memory_plot(IndexClass, index_map)
        self.generate_m_nbit_accuracy_plot(IndexClass, index_map)

    def generate_insert_accuracy_plot(self, IndexClass):
        accuracy = []
        labels = []

        insert_vecs = FileHelper().read_fvecs('./siftsmall/siftsmall_learn.fvecs')
        index = IndexClass()
        index.train(self.evaluation_tool.base, 128)
        for i in range(10):
            # Insert 1000 vectors each time
            new_vecs = insert_vecs[i*1000:(i+1)*1000]
            index.insert(new_vecs)
            accuracy.append(self.evaluation_tool.get_accuracy(index, nprobe=8))
            labels.append(f"insert={i*1000}")

        plot = self._generate_plot(range(10), [accuracy], ["ivf"], 
                                   "insert batches (batch size=1000)", 
                                   "accuracy",
                                   "Accuracy under Different insert batches")
        plot.show()
        self._save_plot(plot, "insert_accuracy_plot.png")

    def generate_insert_time_plot(self, IndexClass):
        insert_vecs = FileHelper().read_fvecs('./siftsmall/siftsmall_learn.fvecs')

        methods = ['Flat', 'IVF', 'PQ']
        colors = ['#4C72B0', '#DD8452']
        average_insert_times = []
        average_query_times = [] 
        for i in range(3):
            index = ...
            if i == 0:  # Flat
                index = IndexClass[i]()
                index.train(self.evaluation_tool.base)
            elif i == 1:  # IVF
                index = IndexClass[i]()
                index.train(self.evaluation_tool.base, 128)
            else:  # PQ
                index = IndexClass[i](8)
                index.train(self.evaluation_tool.base, 7)
            average_insert_times.append(self.evaluation_tool._get_average_insert_time(index, insert_vecs))
            if i == 1:
                average_query_times.append(self.evaluation_tool.get_average_query_time(index, nprobe=8))
            else:
                average_query_times.append(self.evaluation_tool.get_average_query_time(index))

        plt.figure(figsize=(8, 6), dpi=100)
        bar_width = 0.35
        x = np.arange(len(methods))
        _ = plt.bar(x - bar_width/2, average_insert_times, color=colors[0], 
                            width=bar_width, label='Insert Time', edgecolor='black')
        _ = plt.bar(x + bar_width/2, average_query_times, color=colors[1],
                            width=bar_width, label='Query Time', edgecolor='black')
        plt.legend(loc='upper right', fontsize=10, framealpha=1)
        plt.title('Average Insertion Time vs Query Time', fontsize=14, pad=20)
        plt.xlabel('Index Type', fontsize=12)
        plt.ylabel('Time(s)', fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)

        plt.show()
        self._save_plot(plt, "insert_time_plot.png")