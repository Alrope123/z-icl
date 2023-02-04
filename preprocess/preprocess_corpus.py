import os
import json
import math
import argparse
import gzip
from queue import Queue

import numpy as np
import nltk
nltk.download('punkt')

# turns out the nltk version needs to be 3.6.7. note here so that we can add in the README later.
from nltk.tokenize import sent_tokenize


def clean(sentence):
    return sentence \
        .replace("@USER", "@user") \
        .replace("<|EMAIL|>", "<|email|>") \
        .replace("<|PHONE_NUMBER|>", "<|phone_number|>") \
        .replace("\\", "\\\\") \
        .replace("<|endoftext|>", "") \
        .strip()


def break_into_sentences(args, corpus):
    # make sure the output directory exists
    if not os.path.exists(args.corpus_dir):
        os.mkdir(args.corpus_dir)
    out_corpus_dir = os.path.join(args.corpus_dir, corpus)
    if not os.path.exists(out_corpus_dir):
        os.mkdir(out_corpus_dir)
    indices_dir = os.path.join(out_corpus_dir, "shard_indices.json")
    data_dir = os.path.join(out_corpus_dir, "data.jsonl")

    line_num = 0
    # read in raw data and output processed data
    print("Reading and preprocessing the dataset {}...".format(corpus))
    with open(os.path.join(args.corpus_dir, "{}.txt".format(corpus)), 'r') as fin, open(data_dir, 'w') as fout:
        if corpus == "1b":
            for line in fin:
                line = clean(line)
                fout.write(json.dumps({"text": line}))
                fout.write('\n')
                line_num += 1
        elif corpus == "gutenberg":
            current_paragraph = ""
            for line in fin:
                # Find the end of an paragraph
                if line == '\n' and current_paragraph != "" and current_paragraph.replace(".", "").strip()!="":
                    sentences = sent_tokenize(current_paragraph)
                    for sent in sentences:
                        fout.write(json.dumps({"text": sent}))
                        fout.write('\n')
                    line_num += len(sentences)
                    current_paragraph = ""
                elif line != '\n':
                    line = clean(line)
                    # append to the current paragraph
                    if current_paragraph == "":
                        current_paragraph = line
                    else:
                        current_paragraph += " " + line
        elif corpus == "realnews":
            for line in fin:
                line = clean(line)
                fout.write(json.dumps({"text": line}))
                fout.write('\n')
                line_num += 1
        elif corpus == "reviews":
            current_paragraph = ""
            for line in fin:
                if line != '\n':
                    # Find the end of an paragraph
                    index = line.find("<|endoftext|>")
                    if index >= 0 and current_paragraph != "":
                        fout.write(json.dumps({"text": current_paragraph}))
                        fout.write('\n')
                        line_num += 1
                        current_paragraph = ""
                    line = clean(line)
                    # append to the current paragraph
                    if current_paragraph == "":
                        current_paragraph = line
                    else:
                        current_paragraph += " " + line
        elif corpus == "tweets":
            current_paragraph = ""
            for line in fin:
                if line != '\n':
                    # Find the end of an paragraph
                    index = line.find("<|endoftext|>")
                    if index >= 0 and current_paragraph != "":
                        fout.write(json.dumps({"text": current_paragraph}))
                        fout.write('\n')
                        line_num += 1
                        current_paragraph = ""
                    line = clean(line)
                    # append to the current paragraph
                    if current_paragraph == "":
                        current_paragraph = line
                    else:
                        current_paragraph += " " + line
        elif corpus == "yelp":
            current_paragraph = ""
            for line in fin:
                if line != '\n':
                    # Find the end of an paragraph
                    index = line.find("<|endoftext|>")
                    if index >= 0 and current_paragraph != "":
                        fout.write(json.dumps({"text": current_paragraph}))
                        fout.write('\n')
                        line_num += 1
                        current_paragraph = ""
                    line = clean(line)
                    # append to the current paragraph
                    if current_paragraph == "":
                        current_paragraph = line
                    else:
                        current_paragraph += " " + line
        elif corpus == "aclpapers":
            for line in fin:
                line = clean(line)
                # break down into setences
                sentences = sent_tokenize(line)
                for sent in sentences:
                    fout.write(json.dumps({"text": sent}))
                    fout.write('\n')
                line_num += len(sentences)
        elif corpus == "breakingnews":
            for line in fin:
                if line != '\n':
                    line = clean(line)
                    sentences = sent_tokenize(line)
                    # break down into sentences
                    for sent in sentences:
                        fout.write(json.dumps({"text": sent}))
                        fout.write('\n')
                    line_num += len(sentences)
        elif corpus == "contracts":
            for line in fin:
                if line != '\n':
                    line = clean(line)
                    sentences = sent_tokenize(line)
                    # break down into sentences
                    for sent in sentences:
                        fout.write(json.dumps({"text": sent}))
                        fout.write('\n')
                    line_num += len(sentences)
        elif corpus == "cord19":
            for line in fin:
                line = clean(line)
                sentences = sent_tokenize(line)
                # break down into sentences
                for sent in sentences:
                    fout.write(json.dumps({"text": sent}))
                    fout.write('\n')
                line_num += len(sentences)
        elif corpus == "cs":
            for line in fin:
                line = clean(line)
                sentences = sent_tokenize(line)
                # break down into sentences
                for sent in sentences:
                    fout.write(json.dumps({"text": sent}))
                    fout.write('\n')
                line_num += len(sentences)
        elif corpus == "github":
            for line in fin:
                if line != '\n':
                    line = clean(line)
                    sentences = sent_tokenize(line)
                    # break down into sentences
                    for sent in sentences:
                        fout.write(json.dumps({"text": sent}))
                        fout.write('\n')
                    line_num += len(sentences)
        elif corpus == "legal":
            for line in fin:
                if line.find("<|endoftext|>") < 0:
                    line = clean(line)
                    fout.write(json.dumps({"text": line}))
                    fout.write('\n')
                    line_num += 1
        elif corpus == "med":
            for line in fin:
                line = clean(line)
                sentences = sent_tokenize(line)
                # break down into sentences
                for sent in sentences:
                    fout.write(json.dumps({"text": sent}))
                    fout.write('\n')
                line_num += len(sentences)
        elif corpus == "reddit":
            for line in fin:
                line = clean(line)
                fout.write(json.dumps({"text": line}))
                fout.write('\n')
                line_num += 1
        elif corpus == "webtext":
            for line in fin:
                if line != '\n':
                    line = clean(line)
                    fout.write(json.dumps({"text": line}))
                    fout.write('\n')
                    line_num += 1
        elif corpus == "imdb":
            for line in fin:
                if line != '\n':
                    line = clean(line)
                    fout.write(json.dumps({"text": line}))
                    fout.write('\n')
                    line_num += 1
        else:
            for line in fin:
                if line != '\n':
                    line = clean(line)
                    sentences = sent_tokenize(line)
                    for sent in sentences:
                        fout.write(json.dumps({"text": sent}))
                        fout.write('\n')
                    line_num += len(sentences)

    # randomly select data
    np.random.seed(args.seed)
    if line_num >= args.num_shards * args.n_sentence:
        indices = np.random.choice(line_num, (args.num_shards, args.n_sentence), False).tolist()
    else:
        num_shards = min(args.num_shards, math.ceil(line_num / args.n_sentence))
        indices = np.random.permutation(line_num).tolist()
        indices = np.array(indices[:(num_shards-1)*args.n_sentence]).reshape(num_shards-1, args.n_sentence).tolist() + \
            [indices[(num_shards-1)*args.n_sentence:]]

    with open(indices_dir, 'w') as f:
        f.write(json.dumps(indices))
    print("Finished breaking into setences")


def build_text(args, corpus):
    # make sure the output directory exists
    corpus_dir = os.path.join(args.corpus_dir, corpus)
    indices_dir = os.path.join(corpus_dir, "shard_indices.json")
    data_dir = os.path.join(corpus_dir, "data.jsonl")
    assert os.path.exists(corpus_dir), corpus_dir
    assert os.path.exists(indices_dir), indices_dir
    assert os.path.exists(data_dir), data_dir

    # print("Reading the shard indices from file....")
    with open(indices_dir, 'r') as f:
        indices = json.loads(f.readline())
    indices = [set(ids) for ids in indices]
    idx_set = indices[0]
    # acquire the data
    data = []
    with open(data_dir, 'rt') as f:
        for i, line in enumerate(f):
            if i in idx_set:
                data.append(line)

    # output the shards
    with open(os.path.join(corpus_dir, "text.json"), 'w') as f:
        for line in data:
            f.write(line)
    print("Finished producing text")


def build_neighbor(args, corpus):
    # make sure the output directory exists
    corpus_dir = os.path.join(args.corpus_dir, corpus)
    indices_dir = os.path.join(corpus_dir, "shard_indices.json")
    data_dir = os.path.join(corpus_dir, "data.jsonl")
    assert os.path.exists(corpus_dir), corpus_dir
    assert os.path.exists(indices_dir), indices_dir
    assert os.path.exists(data_dir), data_dir

    length = 3

    # print("Reading the shard indices from file....")
    with open(indices_dir, 'r') as f:
        indices = json.loads(f.readline())
    indices = [set(ids) for ids in indices]

    # writing each shards
    # print("Writing each shards...")
    idx_set = indices[0]
    # acquire the data
    first_data = []
    data = []
    data_in_front = []
    data_in_process = []
    q = Queue(1)
    with open(data_dir, 'rt') as f:
        for i, line in enumerate(f):
            line = json.loads(line)["text"]
            finished_data = set()
            for j, data_in_process_entry in enumerate(data_in_process):
                data_in_process_entry.append(line)
                if len(data_in_process_entry) == length:
                    data.append(data_in_process_entry)
                    finished_data.add(j)
            data_in_process = [entry for j, entry in enumerate(data_in_process) if j not in finished_data]
            for j, data_in_front_entry in enumerate(data_in_front):
                if data_in_front_entry["n_left"] > 0:
                    data_in_front_entry["lines"].append(line)
                    data_in_front_entry["n_left"] -= 1
            if q.full():
                if i in idx_set:
                    current_lines = list(q.queue)
                    current_lines.append(line)
                    data_in_process.append(current_lines)
                q.get()
            else:
                if i in idx_set:
                    current_lines = list(q.queue)
                    current_lines.append(line)
                    data_in_front.append({"n_left":1, "lines": current_lines})
                first_data.append(line)
            q.put(line)
    last_data = list(q.queue)
    

    for entry in reversed(data_in_front):
        entry = entry["lines"]
        assert length > len(entry)
        for i in range(length - len(entry)):
            entry.insert(0, last_data[-i-1])
        data.insert(0, entry)
    
    for entry in data_in_process:
        assert length > len(entry)
        for i in range(length - len(entry)):
            entry.append(first_data[i])
        data.append(entry)

    # output the shards
    with open(os.path.join(corpus_dir, "neighbors.jsonl"), 'w') as f:
        for line in data:
            f.write(json.dumps(line))
            f.write("\n")
    print("Finished process corpus {}".format(corpus))


def remove_duplicates(args, corpus):
    # Get path
    corpus_dir = os.path.join(args.corpus_dir, corpus)

    # Read in sentences
    shard_path = os.path.join(corpus_dir, "text.json")
    assert os.path.exists(shard_path), shard_path
    print("Reading corpus shard...")
    sentence_set = set()
    sentences = []
    with open(shard_path, 'r') as f:
        for line in f:
            sents = sent_tokenize(json.loads(line)["text"])
            for sent in sents:
                if sent not in sentence_set:
                    sentence_set.add(sent)
                    sentences.append(sent)

    # output sentences
    with open(shard_path, 'w') as f:
        for sentence in sentences:
            f.write(json.dumps({"text": sentence}))
            f.write('\n')

    print("Finished filtering for {}".format(corpus))


def main(args):
    corpuses = args.corpus.split(',')
    for corpus in corpuses:
        break_into_sentences(args, corpus)
        build_text(args, corpus)
        build_neighbor(args, corpus)
        remove_duplicates(args, corpus)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default="corpuses")
    parser.add_argument("--corpus", type=str, default="1b,cs,legal,med,webtext,realnews," + 
                                                        "reddit,reviews,aclpapers,breakingnews," + 
                                                        "contracts,cord19,github,gutenberg,tweets,yelp")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_shards", type=int, default=100)
    parser.add_argument("--n_sentence", type=int, default=100000)

    args = parser.parse_args()

    main(args)