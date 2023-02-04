# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import argparse

from preprocess_util import load_configs, preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action='store_true',
                    help="Verify the datafiles with pre-computed MD5")
parser.add_argument('--do_test', action='store_true',
                    help="Run 2 tasks per process to test the code")

args = parser.parse_args()

do_train = args.do_train
do_test = args.do_test
if args.do_train and args.do_test:
    raise NotImplementedError("You should specify one of `--do_train` and `--do_test`, not both")
if not args.do_train and not args.do_test:
    raise NotImplementedError("You should specify one of `--do_train` and `--do_test`")

config_dict = load_configs()

class FewshotGymDataset():

    def get_map_hf_dataset_to_list(self):
        return None

    def get_train_test_lines(self, dataset):
        train_lines = [e for e in dataset['train']]
        test_lines = [e for e in dataset['validation']]
        return train_lines, test_lines

    def save(self, path, k, seed, k_shot_train, k_shot_dev, k_shot_test):
        # save to path
        
        config = config_dict[self.hf_identifier]
        k_shot_train = [preprocess(self.hf_identifier, example, config) for example in k_shot_train]
        if do_test:
            k_shot_dev = [preprocess(self.hf_identifier, example, config) for example in k_shot_dev]
            k_shot_test = [preprocess(self.hf_identifier, example, config) for example in k_shot_test]

        if path:
            os.makedirs(os.path.join(path, self.hf_identifier), exist_ok=True)
            prefix = os.path.join(path, self.hf_identifier,
                                    "{}_{}_{}".format(self.hf_identifier, k, seed))
            self.write(k_shot_train, prefix + "_train.jsonl")
            if do_test:
                self.write(k_shot_dev, prefix + "_dev.jsonl")
                self.write(k_shot_test, prefix + "_test.jsonl")

    def write(self, lst, out_file):
        with open(out_file, "w") as fout:
            for line in lst:
                if line is not None:
                    fout.write(line+"\n")

class FewshotGymClassificationDataset(FewshotGymDataset):

    def generate_k_shot_data(self, k, seed, path=None):
        """
        generate a k-shot (k) dataset using random seed (seed)
        return train, dev, test
        """


        if self.hf_identifier not in config_dict:
            return None, None, None

        if do_train:
            if seed<100:
                return None, None, None
            k = 16384
        elif do_test:
            k = 16
            path = "datasets"

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        train_lines, test_lines = self.get_train_test_lines(dataset)

        # shuffle the data
        np.random.seed(seed)
        np.random.shuffle(train_lines)

        # Get label list for balanced sampling
        label_list = {}
        for line in train_lines:
            label = "all"
            if label not in label_list:
                label_list[label] = [line]
            else:
                label_list[label].append(line)

        # make train, dev, test data
        k_shot_train = []
        for label in label_list:
            for line in label_list[label][:k]:
                k_shot_train.append(line)

        k_shot_dev = []
        for label in label_list:
            for line in label_list[label][k:2*k]:
                k_shot_dev.append(line)

        k_shot_test = test_lines

        # save to path
        self.save(path, k, seed, k_shot_train, k_shot_dev, k_shot_test)
        return k_shot_train, k_shot_dev, k_shot_test
