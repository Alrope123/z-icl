import os
import argparse
import json
import numpy as np
import sys

sys.path.insert(1, os.path.abspath("utils"))
from util import get_variant_datasets, translate_dataset_name
from templates import create_version_w_template

LABLE_MAPPINGS = {
    "SST-2": {
        "terrible": "bad",
        "great": "good"
    },
    "amazon_polarity": {
        "negative": "bad",
        "positive": "good"
    },
    "amazon": {
        "terrible": "horrible",
        "bad": "negative",
        "okay": "neutral",
        "good": "positive",
        "great": "excellent"
    },
    "yelp_polarity": {
        "negative": "bad",
        "positive": "good"
    },
    "yelp_full": {
        "terrible": "horrible",
        "bad": "negative",
        "okay": "neutral",
        "good": "positive",
        "great": "excellent"
    },
    "tweet_eval-hate": {
        "negative": "bad",
        "neutral": "normal",
        "positive": "good"
    },
    "mr": {
        "terrible": "bad",
        "great": "good"
    },
    "SST-2": {
        "terrible": "bad",
        "great": "good"
    },
    "sst-5": {
        "terrible": "horrible",
        "bad": "negative",
        "okay": "neutral",
        "good": "positive",
        "great": "excellent"
    },
}


def main(args):
    assert args.method in ["nearest", "dn", "pn", "ri"]

    datasets = get_variant_datasets(translate_dataset_name(args.dataset))
    for dataset in datasets:
        if dataset.endswith("-n=500") and args.method == "nearest":
            continue

        new_dataset = dataset + "_" + args.method + \
                        ("_synm" if args.synonym_labeling else "")

        new_dataset_dir = os.path.join(args.data_dir, new_dataset)
        if not os.path.exists(new_dataset_dir):
            os.mkdir(new_dataset_dir)

        print("Retrieving the sentences...")
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
                    dp["task"] = new_dataset
                    test_data.append(dp)
            with open(os.path.join(new_dataset_dir, "{}_{}_{}_{}.jsonl".format(new_dataset, args.k, seed, "test")), "w") as f:
                for dp in test_data:
                    f.write(json.dumps(dp))
                    f.write("\n")

            print("Aggregate actual data...")
            # build train data
            train_data = []
            retrieved_data_path = os.path.join("retrieved", "{}_{}".format(dataset, args.method), 
                                                "{}_{}_{}_{}.jsonl".format(dataset, args.method, args.k, seed))
            with open(retrieved_data_path, "r") as f:
                for line in f:
                    train_line = []
                    line = json.loads(line)
                    for s in line:
                        dp = {} 
                        dp["task"] = new_dataset
                        dp["input"] = s
                        dp["output"] = np.random.choice(test_data[0]["options"])
                        dp["options"] = test_data[0]["options"]
                        if args.synonym_labeling:
                            original_dataset = dataset.split('-n=')[0]
                            dp["output"] = LABLE_MAPPINGS[original_dataset][dp["output"]]
                            dp["options"] = [LABLE_MAPPINGS[original_dataset][option] for option in dp["options"]]
                        train_line.append(dp)
                    train_data.append(train_line)

            # write the modified data
            with open(os.path.join(new_dataset_dir, "{}_{}_{}_{}.jsonl".format(new_dataset, args.k, seed, "train")), "w") as f:
                for dp in train_data:
                    f.write(json.dumps(dp))
                    f.write("\n")

        create_version_w_template(args, new_dataset)


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--method", type=str, default="nearest")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")
    parser.add_argument("--synonym_labeling", default=False, action="store_true")

    parser.add_argument("--data_dir", type=str, default="datasets")

    args = parser.parse_args()

    main(args)