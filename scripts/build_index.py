import os
import io
import argparse
import json
import numpy as np
import faiss
import sys
from nltk.tokenize import sent_tokenize
sys.path.insert(1, os.path.abspath("utils"))
from util import build_embeddings

def main(args):
    corpuses = args.corpus.split(',')

    print("Reading...")
    for corpus in corpuses:
        build_corpus_embeddings(args=args, corpus=corpus)
        build_neighbor_dict(args=args, corpus=corpus)
    combine_corpuses(args=args, corpuses=corpuses)
    build_corpus_index(args=args)


def build_corpus_embeddings(args, corpus):
    # Get path
    out_dir = os.path.join(args.corpus_dir, corpus)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Read in sentences
    shard_path = os.path.join(out_dir, "text.json")
    assert os.path.exists(shard_path), shard_path
    print("Reading corpus shard from {}".format(shard_path))
    sentences = []
    with open(shard_path, 'r') as f:
        for line in f:
            sentences.append(json.loads(line)["text"])
    build_embeddings(os.path.join(out_dir, "embeddings.npy"), sentences, args.batch_size)
    print("Finished building embeddings for corpus {}".format(corpus))


def build_neighbor_dict(args, corpus):
    # make sure the output directory exists
    corpus_dir = os.path.join(args.corpus_dir, corpus)
    assert os.path.exists(corpus_dir), corpus_dir
    neighbor_in_path = os.path.join(corpus_dir, "neighbors.jsonl")
    assert os.path.exists(neighbor_in_path), neighbor_in_path
    out_dict_path = os.path.join(corpus_dir, "dict.json")
    
    print("Reading the shard indices from file....")
    
    with open(neighbor_in_path, 'r') as f:
        lines = f.readlines()
    print("Finished reading the units")

    dict = {}
    for line in lines:
        units = json.loads(line)
        center_sentences = sent_tokenize(units[1]) if units[1] != "" else [units[1]]
        front_sentences = sum([sent_tokenize(sent) if sent != "" else [sent] for sent in units[:1]], [])
        tail_sentences = sum([sent_tokenize(sent) if sent != "" else [sent] for sent in units[1+1:]], [])
        center_sentences_indices = [i + len(front_sentences) for i in range(len(center_sentences))]
        all_sentences = sum([front_sentences, center_sentences, tail_sentences], [])
        for idx in center_sentences_indices:
            neighbors = {i: all_sentences[idx+i] for i in range(-1, 1+1)}
            del neighbors[0]
            dict[all_sentences[idx]] = neighbors

    # output sentences
    with open(out_dict_path, 'w') as f:
        f.write(json.dumps(dict))

    print("Finished building dict for {}".format(corpus)) 


def combine_corpuses(args, corpuses):
    # Concatenate the embeddings of different corpuses
    vectors = []
    for corpus in corpuses:
        embed_dir = os.path.join(args.corpus_dir, corpus, "embeddings.npy")
        assert embed_dir, embed_dir
        vectors.append(np.load(embed_dir))
    
    vectors = np.concatenate(vectors, axis=0)

    print("Outputing...")
    # # Output the combined embeddings
    new_corpus_dir = os.path.join(args.corpus_dir, args.new_corpus_name)
    if not os.path.exists(new_corpus_dir):
        os.mkdir(new_corpus_dir)
    new_embed_path = os.path.join(new_corpus_dir, "embeddings.npy")
    np.save(new_embed_path, vectors)


def build_corpus_index(args):
    def build_index(index_path, corpus_vec_path):
        corpus_vec = np.load(corpus_vec_path)
        corpus_vec = corpus_vec.astype(np.float32)
        print(corpus_vec.shape)
        index = faiss.IndexFlatIP(corpus_vec.shape[1])
        index.add(corpus_vec)
        faiss.write_index(index, index_path)

    # Get path
    method_dir = os.path.join(args.corpus_dir, args.new_corpus_name)
    embed_path = os.path.join(method_dir, "embeddings.npy")
    index_path = os.path.join(method_dir, "indices.IndexFlatIP")

    # make sure there's embedding
    assert os.path.exists(embed_path), embed_path
    print("Reading the embedding from " + embed_path)

    print("Building index...")
    # build index
    build_index(index_path, embed_path)
    print("Finished building the index for corpus {}".format(args.new_corpus_name))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default="corpuses")
    parser.add_argument("--corpus", type=str, default="1b,cs,legal,med,webtext,realnews," + 
                                                        "reddit,reviews,aclpapers,breakingnews," + 
                                                        "contracts,cord19,github,gutenberg,tweets,yelp")
    parser.add_argument("--new_corpus_name", type=str, default="ours")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    main(args)