# SST-2 amazon_polarity_no_prefix cr mr tweet_eval-sentiment yelp_polarity-n=2000 amazon-n=2000 sst-5-n=2000 yelp_full-n=2000

import os
import argparse
import json
import numpy as np
import sys
import csv

sys.path.insert(1, os.path.abspath("utils"))
from templates import create_version_w_template
from util import GPT3_DATASETS, OVERSIZED_DATASETS, translate_dataset_name, get_variant_datasets


def get_labels(dataset):
    if dataset in ["SST-2", "mr", "cr"]:
        label_words = ["terrible", "great"]
    elif dataset in ["sst-5", "yelp_full", "amazon"]:
        label_words = ["terrible", "bad", "okay", "good", "great"]
    else:
        raise NotImplementedError(dataset)
    return label_words


def transform_dataset(args, dataset):
    data_dir = os.path.join("data-for-public", "k-shot")
    seeds = args.seed.split(",")
    k = args.k
    
    output_dir = os.path.join(args.data_dir, dataset)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for seed in seeds:
        input_dir = os.path.join(data_dir, dataset, "{}-{}".format(k, seed))
        output_filename = os.path.join(output_dir, "{}_{}_{}".format(dataset, k, seed))                
        # to see which format is the dataset 
        is_tsv = True if dataset in ["amazon",  "SST-2", "yelp_full"] else False
        labels = get_labels(dataset)
        train_data = []
        test_data = []
        
        # read in 
        if is_tsv:
            # read in train set
            with open(os.path.join(input_dir, "train.tsv"), 'r') as f: 
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    chunks = line.split('\t')
                    train_data.append({
                        "task": dataset,
                        "input": chunks[0],
                        "output": labels[int(chunks[1])],
                        "options": labels
                    })
            # read in test set
            with open(os.path.join(data_dir, dataset, "{}-100".format(k), "test.tsv"), 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    chunks = line.split('\t')
                    test_data.append({
                        "task": dataset,
                        "input": chunks[0],
                        "output": labels[int(chunks[1])],
                        "options": labels
                    })
        else:
            # read in train set
            with open(os.path.join(input_dir, "train.csv"), 'r') as f:
                for label, text in csv.reader(f):
                    train_data.append({
                        "task": dataset,
                        "input": text,
                        "output": labels[int(label)],
                        "options": labels
                    })
            with open(os.path.join(data_dir, dataset, "{}-100".format(k), "test.csv"), 'r') as f:
                for label, text in csv.reader(f):
                    test_data.append({
                        "task": dataset,
                        "input": text,
                        "output": labels[int(label)],
                        "options": labels
                    })

        # output
        with open("{}_train.jsonl".format(output_filename), 'w') as f:
            for line in train_data:
                f.write(json.dumps(line))
                f.write('\n')
        with open("{}_test.jsonl".format(output_filename), 'w') as f:
            for line in test_data:
                f.write(json.dumps(line))
                f.write('\n')


def random_label(args, dataset):
    new_dataset = dataset + "_random" 
    new_dataset_dir = os.path.join(args.data_dir, new_dataset)
    if not os.path.exists(new_dataset_dir):
        os.mkdir(new_dataset_dir)

    seeds = args.seed.split(',')
    for seed in seeds:
        # random seed
        np.random.seed(int(seed))

        train_data = []
        train_data_path = os.path.join(args.data_dir, dataset, "{}_{}_{}_{}.jsonl".format(dataset, args.k, seed, "train"))
        with open(train_data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                assert dp["task"]==dataset
                dp["task"] = new_dataset
                train_data.append(dp)

        test_data = []
        test_data_path = os.path.join(args.data_dir, dataset, "{}_{}_{}_{}.jsonl".format(dataset, args.k, seed, "test"))
        with open(test_data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                assert dp["task"]==dataset
                dp["task"] = new_dataset
                test_data.append(dp)
        
        for dp in train_data:
            dp["output"] = dp["options"][np.random.choice(range(len(dp["options"])))]


        with open(os.path.join(new_dataset_dir, "{}_{}_{}_{}.jsonl".format(new_dataset, args.k, seed, "train")), "w") as f:
                for dp in train_data:
                    f.write(json.dumps(dp))
                    f.write("\n")

        with open(os.path.join(new_dataset_dir, "{}_{}_{}_{}.jsonl".format(new_dataset, args.k, seed, "test")), "w") as f:
            for dp in test_data:
                f.write(json.dumps(dp))
                f.write("\n")

    

def remove_prefix(args, dataset):
    seeds = args.seed.split(',')
    for seed in seeds:
        # readin the test data and output it as it is
        test_data = []
        test_data_path = os.path.join(args.data_dir, dataset, "{}_{}_{}_{}.jsonl".format(dataset, args.k, seed, "test"))
        with open(test_data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                assert dp["task"]==dataset
                if not dp["input"].startswith("title:"):
                    return
                dp["input"] = dp["input"][7:]
                test_data.append(dp)
        with open(os.path.join(test_data_path), "w") as f:
            for dp in test_data:
                f.write(json.dumps(dp))
                f.write("\n")

        # build train data
        train_data = []
        train_data_path = os.path.join(args.data_dir, dataset, "{}_{}_{}_{}.jsonl".format(dataset, args.k, seed, "train"))
        with open(train_data_path, "r") as f:
            for line in f: 
                dp = json.loads(line)
                assert dp["task"] == dataset
                dp["input"] = dp["input"][7:]
                train_data.append(dp)


        # write the modified data
        with open(train_data_path, "w") as f:
            for dp in train_data:
                f.write(json.dumps(dp))
                f.write("\n")


def down_size_dataset(args, dataset, n):
    np.random.seed(100)

    new_dataset = dataset + "-n={}".format(n) 
    new_dataset_dir = os.path.join(args.data_dir, new_dataset)
    if not os.path.exists(new_dataset_dir):
        os.mkdir(new_dataset_dir)

    test_data = []
    test_data_path = os.path.join(args.data_dir, dataset, "{}_{}_{}_{}.jsonl".format(dataset, args.k, 100, "test"))
    with open(test_data_path, "r") as f:
        for line in f:
            dp = json.loads(line)
            assert dp["task"]==dataset
            dp["task"] = new_dataset
            test_data.append(dp)
    
    # count labels
    label_indices = {}
    for i, dp in enumerate(test_data):
        if dp["output"] in label_indices:
            label_indices[dp["output"]].append(i) 
        else:
            label_indices[dp["output"]] = [i]
    n_labels =  len(label_indices.keys())
    n_entry = sum([len(v) for k, v in label_indices.items()])
    n_per_label = {k: int(n / n_entry * len(v)) for k, v in label_indices.items()}
    
    # fill up remainding test inputs
    diff = n - sum(n_per_label.values())
    if diff > 0:
        for i, (k, v) in enumerate(n_per_label.items()):
            if i >= diff:
                break
            n_per_label[k] +=1 
    assert n - sum(n_per_label.values()) == 0

    selected_indices = []
    # randomly select indices
    for label, indices in label_indices.items():
        selected_indices.extend(np.random.choice(indices, n_per_label[label], replace=False))

    test_data = [dp for i, dp in enumerate(test_data) if i in selected_indices]
    np.random.shuffle(test_data)

    seeds = args.seed.split(',')
    for seed in seeds:
        train_data = []
        train_data_path = os.path.join(args.data_dir, dataset, "{}_{}_{}_{}.jsonl".format(dataset, args.k, seed, "train"))
        with open(train_data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                assert dp["task"]==dataset
                dp["task"] = new_dataset
                train_data.append(dp)

        # write the modified data
        with open(os.path.join(new_dataset_dir, "{}_{}_{}_{}.jsonl".format(new_dataset, args.k, seed, "train")), "w") as f:
            for dp in train_data:
                f.write(json.dumps(dp))
                f.write("\n")

        with open(os.path.join(new_dataset_dir, "{}_{}_{}_{}.jsonl".format(new_dataset, args.k, seed, "test")), "w") as f:
            for dp in test_data:
                f.write(json.dumps(dp))
                f.write("\n")


def main(args):

    datasets = args.dataset.split(',')

    for dataset in datasets:
        dataset = translate_dataset_name(dataset)
        transform = dataset in ["SST-2", "mr", "cr", "sst-5", "yelp_full", "amazon"]
        prefix = dataset == "amazon_polarity"
        gpt3 = dataset in GPT3_DATASETS
        oversize = dataset in OVERSIZED_DATASETS

        if transform:
            transform_dataset(args, dataset)
        if prefix:
            remove_prefix(args, dataset)
        if gpt3:
            down_size_dataset(args, dataset, 500)
        if oversize:
            down_size_dataset(args, dataset, 2000)
        print("Finished formatting {}.".format(dataset))
    
        for dataset in get_variant_datasets(dataset):
            random_label(args, dataset)
            create_version_w_template(args, dataset)
            create_version_w_template(args, dataset + "_random")
            print("Finished creating variations for {}.".format(dataset))
    
    print("Done modifying all datasets.")


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="SST2,Amz,CR,MR,Tweet,Yelp,Amz5,SST5,Yelp5")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")

    parser.add_argument("--data_dir", type=str, default="datasets")

    args = parser.parse_args()

    main(args)