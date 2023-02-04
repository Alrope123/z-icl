# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json

GPT3_DATASETS = ["cr", "amazon_polarity", "yelp_polarity", "tweet_eval-sentiment", "SST-2"]
OVERSIZED_DATASETS = ["yelp_polarity", "amazon", "sst-5", "yelp_full"]


def load_data(task, split, k, dataset_dir, seed=0):
    dataset = task

    data = []
    data_path = os.path.join(dataset_dir, dataset,
                                "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
    with open(data_path, "r") as f:
        for line in f:
            dp = json.loads(line)
            data.append(dp)
    return data


def build_embeddings(save_dir, sentences, batch_size=64):
    import numpy as np
    import torch
    from simcse import SimCSE
    
    if save_dir != None and os.path.exists(save_dir):
        return

    print("building the embeddings....")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get embeddings
    model = SimCSE("princeton-nlp/unsup-simcse-bert-large-uncased")
    embeddings = model.encode(sentences, device=device, return_numpy=True, batch_size=batch_size, max_length=256).tolist()

    if save_dir != None:
        np.save(save_dir, embeddings)
    else:
        return np.array(embeddings)


def translate_dataset_name(dataset):
    return {
        "CR": "cr",
        "Amz": "amazon_polarity", 
        "Amz5": "amazon",
        "Yelp": "yelp_polarity",
        "Yelp5": "yelp_full",
        "Tweet": "tweet_eval-sentiment",
        "MR": "mr",
        "SST2": "SST-2",
        "SST5": "sst-5"
    }[dataset]

def get_variant_datasets(dataset):
    datasets = []
    if dataset in GPT3_DATASETS:
        datasets.append(dataset + "-n=500")
    if dataset in OVERSIZED_DATASETS:
        datasets.append(dataset + "-n=2000")
    else:
        datasets.append(dataset)
    return datasets