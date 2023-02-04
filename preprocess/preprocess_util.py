# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import string


def load_configs():
    config_dict = {}
    for task in os.listdir("config"):
        with open(os.path.join("config", task), "r") as f:
            config = json.load(f)
        config_dict[task.split(".")[0]] = config
    return config_dict


def preprocess(dataset, line, config):
    input_, output_ = line
    input_ = input_.strip().replace("\\n", " ")
    output_ = str(output_).split("\t")[0].strip()

    if dataset=="superglue-multirc" and output_=="NO ANSWER!":
        return None

    do_handle_sep = dataset.startswith("race-") or \
            dataset in ["sciq", "social_i_qa", "wiqa", "quail",
                        "superglue-multirc"]

    if do_handle_sep:
        assert input_.count("[SEP]")==1
        input_, context = input_.split("[SEP]")

    alphabet_options = list(string.ascii_uppercase)
    if dataset in ["quail", "quarel"]:
        alphabet_options = ["(" + option + ")" for option in alphabet_options]
    else:
        alphabet_options = [" (" + option + ") " for option in alphabet_options]

    options = []

    assert len(config["options"])>=2
    options = config["options"]

    if do_handle_sep:
        input_ = context + input_

    return json.dumps({"task": dataset, "input": input_, "output": output_, "options": options})




