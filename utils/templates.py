import string
import os
import json
import numpy as np
from copy import deepcopy

TEMPLATES = ("Review: {}", "Sentiment: {}")

def apply_template(dp):
    dp = deepcopy(dp)
    dp["input"] = TEMPLATES[0].format(dp["input"])
    dp["output"] = TEMPLATES[1].format(dp["output"])
    for i, option in enumerate(dp["options"]):
        dp["options"][i] =TEMPLATES[1].format(option)
    return dp


def create_version_w_template(args, dataset):
    new_dataset = dataset + "_t" 
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
                line = json.loads(line)
                if type(line) is dict:
                    dp = line
                    assert dp["task"]==dataset
                    dp["task"] = new_dataset
                    train_data.append(apply_template(dp))
                else:
                    new_line = []
                    for dp in line:
                        assert dp["task"]==dataset
                        dp["task"] = new_dataset
                        new_line.append(apply_template(dp))
                    train_data.append(new_line)

        test_data = []
        test_data_path = os.path.join(args.data_dir, dataset, "{}_{}_{}_{}.jsonl".format(dataset, args.k, seed, "test"))
        with open(test_data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                assert dp["task"]==dataset
                dp["task"] = new_dataset
                test_data.append(apply_template(dp))

        with open(os.path.join(new_dataset_dir, "{}_{}_{}_{}.jsonl".format(new_dataset, args.k, seed, "train")), "w") as f:
                for dp in train_data:
                    f.write(json.dumps(dp))
                    f.write("\n")

        with open(os.path.join(new_dataset_dir, "{}_{}_{}_{}.jsonl".format(new_dataset, args.k, seed, "test")), "w") as f:
            for dp in test_data:
                f.write(json.dumps(dp))
                f.write("\n")