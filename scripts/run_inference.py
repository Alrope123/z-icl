# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pickle as pkl
import logging
import numpy as np
import time
import sys

from transformers import AutoTokenizer, GPTNeoXTokenizerFast

sys.path.insert(1, os.path.abspath("utils"))
sys.path.insert(1, os.path.abspath("ICL"))
from data import Data
from model import Model
from util import load_data, translate_dataset_name, GPT3_DATASETS, OVERSIZED_DATASETS


def main(logger, args):
    assert args.dataset is not None

    dataset = get_dataset_name(args)

    if "gpt-3" in args.gpt:
        assert args.dataset in GPT3_DATASETS
        assert args.variant in ["z-icl", "no_demos", "icl-gold", "icl-random"], "other variant not supported for GPT-3."

    if "neox" in args.gpt:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        

    ### checkpoint ...
    icl_model = Model(logger, args.out_dir)

    # setup hyperparams for data
    max_length_per_example = 256
    max_length = 256
    if not args.variant == "no_demos":
        max_length = min(max_length * args.k, 1024)

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    icl_data = Data(logger, tokenizer, args.model_method, not args.variant == "no_demos",
                               args.k, max_length, max_length_per_example)

    results = []
    errors = []
    seeds = args.seed.split(",")

    for seed in seeds:
        ### data ...
        train_data = load_data(dataset, "train", args.k, dataset_dir=args.data_dir, seed=seed)
        dev_data = load_data(dataset, args.split, args.k, dataset_dir=args.data_dir, seed=seed)
        test_task = dev_data[0]["task"]

        macro, acc = run(logger, test_task, icl_data, icl_model,
                        train_data, dev_data, seed, True)

        if macro is None or acc is None:
            errors.append("%s/%s" % (test_task, seed))
        else:
            results.append((macro, acc))

    logger.info("Macro-F1 %s over %d target tasks: %.1f" % (dataset, len(results) // len(seeds), 100*np.mean([r[0] for r in results])))
    logger.info("Variance (Macro-F1)" + " %s over %d target tasks: %.1f" % (dataset, len(results) // len(seeds), 100*np.std([r[0] for r in results])))
    logger.info("Accuracy %s over %d target tasks: %.1f" % (dataset, len(results) // len(seeds), 100*np.mean([r[1] for r in results])))
    logger.info("Variance (Accuacy)" + " %s over %d target tasks: %.1f" % (dataset, len(results) // len(seeds), 100*np.std([r[1] for r in results])))

    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")


def run(logger, task, icl_data, icl_model, train_data, dev_data, seed, is_classification):
    split_name = args.split
    cache_path = os.path.join(args.out_dir,
                                "{}_{}-{}-{}{}{}-{}.pkl".format(
                                    task,
                                    args.variant,
                                    split_name,
                                    icl_data.method,
                                    "-k={}".format(args.k) if not args.variant == "no_demos" else "",
                                    "-s={}".format(seed) if not args.variant == "no_demos" else "",
                                    args.gpt))

    icl_data.tensorize(train_data, dev_data, add_newlines=True)
    icl_data.print_tensorized_example()
    logger.info(cache_path)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            losses = pkl.load(f)
    else:
        if icl_model.is_none():
            icl_model.load(gpt=args.gpt, use_int8=args.use_int8)
            icl_model.to_device()
            icl_model.eval()
        startime = time.time()
        losses = icl_model.do_inference(icl_data, args.test_batch_size, verbose=True)
        endtime = time.time()
        logger.info("Total time spent for inference: {}".format(endtime - startime))
        with open(cache_path, "wb") as f:
            pkl.dump(losses, f)

    assert len(losses)==len(icl_data)

    predictions = icl_model.do_predict(icl_data, losses=losses)
    groundtruths = [dp["output"] for dp in dev_data]
    macro = icl_data.evaluate(predictions, groundtruths, is_classification, report_accuracy=False)
    acc = icl_data.evaluate(predictions, groundtruths, is_classification, report_accuracy=True)
    logger.info("Macro=%s" % macro)
    logger.info("Accuracy=%s" % acc)

    prediction_path = cache_path.replace(".pkl", ".txt")

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            f.write(prediction)
            f.write("\n")

    return macro, acc



def get_dataset_name(args):
    dataset = translate_dataset_name(args.dataset)

    if dataset in OVERSIZED_DATASETS or "gpt-3" in args.gpt:
        n = 500 if "gpt-3" in args.gpt else 2000
        dataset += "-n={}".format(n)

    if args.variant == "z-icl":
        dataset += "_pn_synm"
    elif args.variant == "icl-random":
        dataset += "_random"
    elif args.variant == "naive_z-icl":
        dataset += "_nearest"
    elif args.variant == "random_inputs":
        dataset += "_ri"

    if "neox" not in args.gpt:
        dataset += "_t"
    
    return dataset


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="z-icl", type=str, choices=["z-icl", "no_demos", "icl-gold",
                                                             "icl-random", "naive_z-icl", "random_inputs"])

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")

    parser.add_argument("--test_batch_size", type=int, default=8)

    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--out_dir", type=str, default="out")

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_method", type=str, default="direct", choices=["direct", "channel"])
    parser.add_argument("--gpt", type=str, default="gpt-j", choices=["gpt-j", "gpt-neox"])
    parser.add_argument("--use_int8", default=False, action="store_true")
    

    args = parser.parse_args()

    # DEBUG
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    args.log_file = os.path.join(args.out_dir, "{}-{}-{}-log.txt".format(args.dataset, args.variant, args.model_method))

    # config logger
    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)