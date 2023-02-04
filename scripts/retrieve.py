
import os
import argparse
import json
import numpy as np
import faiss
from tqdm import tqdm
import sys
sys.path.insert(1, os.path.abspath("utils"))
from util import get_variant_datasets, translate_dataset_name, build_embeddings


def faiss_search(query_vec_path, index, K,  k=16, query_vec=[]):
    if len(query_vec) == 0:
        query_vec = np.load(query_vec_path)
    query_vec = query_vec.astype(np.float32)

    D, I = index.search(query_vec, K)
    assert D.shape == I.shape == (query_vec.shape[0], K)
    print("Selecting diverse examples...")
    prediction_indices = I
    if K > k:
        prediction_indices = [np.random.choice(I[i], k, replace=False) for i in range(len(I))]
    return np.flip(D, 1), np.flip(prediction_indices, 1)


def get_source_corpus(source_corpuses, example_idx):
    for upper_bound, source_corpus in source_corpuses.items():
        if example_idx < upper_bound:
            return source_corpus 


def main(args):
    assert args.method in ["nearest", "dn", "pn", "ri"]

    datasets = get_variant_datasets(translate_dataset_name(args.dataset))

    print("Loading the corpus...")
    # read in corpus text data and neighbor dictionary
    corpus_data = []
    corpuses = args.corpus.split(',')
    source_corpuses = {}
    total_count = 0
    for corpus in corpuses:
        corpus_dir = os.path.join(args.corpus_dir, corpus)
        assert os.path.exists(corpus_dir), corpus_dir
        shard_dir = os.path.join(corpus_dir, "text.json")
        assert os.path.exists(shard_dir), shard_dir
        print("Reading corpus data from " + shard_dir)
        count = 0
        with open(shard_dir, 'r') as f:
            for line in f:
                corpus_data.append(json.loads(line)["text"])
                count += 1
        total_count += count
        source_corpuses[total_count] = corpus
    if args.method == "pn":
        neighbor_dict = {}
        for corpus in corpuses:
            corpus_dir = os.path.join(args.corpus_dir, corpus)
            assert os.path.exists(corpus_dir)
            dict_dir = os.path.join(corpus_dir, "dict.json")
            assert os.path.exists(dict_dir), dict_dir
            print("Reading dictionary from " + dict_dir)
            with open(dict_dir, "r") as f:
                neighbor_dict[corpus] = json.load(f)


    # read in corpus index
    index = None
    idx_dir = os.path.join(args.corpus_dir, args.corpus_name, "indices.IndexFlatIP")
    assert os.path.exists(idx_dir), idx_dir
    print("Reading the corpus index from " + idx_dir)
    index = faiss.read_index(idx_dir)

    ################################################################################################################

    print("Retrieving the sentences...")
    for dataset in datasets:
        seeds = args.seed.split(',')
        for seed in seeds:
            # random seed
            np.random.seed(int(seed))

            # readin the test data and output it as it is
            test_data = []
            test_data_path = os.path.join(args.data_dir, dataset, "{}_{}_{}_{}.jsonl".format(dataset, args.k, 100, "test"))
            with open(test_data_path, "r") as f:
                for line in f:
                    dp = json.loads(line)
                    assert dp["task"]==dataset
                    test_data.append(dp)
            
            print("Getting indices of the demonstrations...")
            if args.method == "ri":
                # randomly sample indices
                indices = np.random.choice(len(corpus_data), args.k, replace=False)
                # tile this up
                indices = np.tile(indices, (len(test_data), 1))
                print("Finished selecting random indices...")
            else: 
                # build embeddings and look for nearest neighbors for the test inputs
                query_dir = os.path.join(args.data_dir, dataset, "simcse_embeddings_test.npy")
                if not os.path.exists(query_dir):
                    print("Building the embeddings for the test dataset")
                    build_embeddings(query_dir, [dp["input"] for dp in test_data], args.batch_size)
                print("Finished building embeddings, searching using faiss")
                values, indices = faiss_search(query_dir, index, args.K if args.method == "dn" else args.k, args.k, query_vec=[])

            print("Aggregate actual data...")
            # build train data
            train_data = []
            for i, test_dp in enumerate(tqdm(test_data)):
                train_line = []
                for j in range(args.k):
                    if args.method == "pn": 
                        inp = neighbor_dict[get_source_corpus(source_corpuses, indices[i][j])][corpus_data[indices[i][j]]]["-1"]
                    else:
                        inp = corpus_data[indices[i][j]]
                    train_line.append(inp)
                assert len(train_line) == args.k
                train_data.append(train_line)

            # write the modified data
            if not os.path.exists("retrieved"):
                os.mkdir("retrieved")
            if not os.path.exists(os.path.join("retrieved", "{}_{}".format(dataset, args.method))):
                os.mkdir(os.path.join("retrieved", "{}_{}".format(dataset, args.method)))
            with open(os.path.join("retrieved", "{}_{}".format(dataset, args.method), "{}_{}_{}_{}.jsonl".format(dataset, args.method, args.k, seed)), "w") as f:
                for dp in train_data:
                    f.write(json.dumps(dp))
                    f.write("\n") 


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--corpus", type=str, default="1b,cs,legal,med,webtext,realnews," + 
                                                    "reddit,reviews,aclpapers,breakingnews," + 
                                                    "contracts,cord19,github,gutenberg,tweets,yelp")
    parser.add_argument("--corpus_name", type=str, default="ours")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")
    parser.add_argument("--method", type=str, default="nearest")

    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--corpus_dir", type=str, default="corpuses")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    main(args)