import sys, os, getopt, time, glob, collections
import marisa_trie
from bs4 import BeautifulSoup
import lxml, cchardet
import opencc
from nltk import flatten, bigrams, FreqDist
from nltk.lm.preprocessing import pad_both_ends
import pandas as pd

mode = "real"
mypath = "data/"

opts, args = getopt.getopt(sys.argv[1:], "trp:", ["test", "real", "path="])
for o, a in opts:
    if o in ("-t", "--test"):
        mode = "test"
    if o in ("-r", "--real"):
        mode = "real"
    if o in ("-p", "--path"):
        mypath = a

missing = collections.Counter()
found = collections.Counter()

## Prepare directories
os.system("mkdir -p output; cd output; mkdir -p d-2gram d-words")
logfile = open("output/getfreq.log", "a")


def log(msg):
    print(msg)
    logfile.write(f'{time.strftime("%H:%M:%S", time.localtime())}\t{msg}\n')
    logfile.flush()


def import_data(mode):
    if mode == "test":
        return [
            "食",
            "魚",
            "飯",
            "我",
            "你",
            "去",
            "早餐",
            "午餐",
            "唔",
            "早",
            "好",
            "而家",
            "我哋",
            "魚生",
            "魚生飯",
            "今",
            "早",
            "午",
            "而",
            "佢",
            "日",
            "今日",
        ]

    elif mode == "real":
        data_words_hk = pd.read_csv(
            "source/existingwordcount.csv",
            header=None,
            skiprows=[0],
            names=["char", "freq"],
            sep=",",
        )
        data_essay = pd.read_csv(
            "source/essay.txt", header=None, names=["char", "freq"], sep="\t"
        )
        data_essay_cantonese = pd.read_csv(
            "source/essay-cantonese.txt", header=None, names=["char", "freq"], sep="\t"
        )
        data_jyutpingdict1 = pd.read_csv(
            "source/jyut6ping3.dict.tsv",
            header=None,
            names=["char", "jp", "freq"],
            sep="\t",
        )
        data_jyutpingdict2 = pd.read_csv(
            "source/jyut6ping3.lettered.dict.tsv",
            header=None,
            names=["char", "freq"],
            sep="\t",
        )
        data_jyutpingdict3 = pd.read_csv(
            "source/jyut6ping3.maps.dict.tsv",
            header=None,
            names=["char", "freq"],
            sep="\t",
        )
        log("Imported Dicts")

        merged_words = pd.concat(
            [
                data_essay,
                data_words_hk,
                data_essay_cantonese,
                data_jyutpingdict1,
                data_jyutpingdict2,
                data_jyutpingdict3,
            ]
        )
        log("Merged Dicts")

        words = merged_words.char.tolist()
        words_t2hk = []
        converter = opencc.OpenCC("t2hk.json")
        for w in words:
            w2 = converter.convert(w)
            if w2 != w:
                words_t2hk.append(w2)
        log(f"Added {len(words_t2hk)} HK orthographic vairants.")
        return words + words_t2hk


def getcontent(f):
    with open(f, "r") as file:
        s = file.read()
        soup = BeautifulSoup(s, "lxml")
        return soup.get_text()


def update_fdist(date, parsed_sentences):
    tokens = flatten([list(pad_both_ends(sent, n=2)) for sent in parsed_sentences])
    fdist_tokens = FreqDist(tokens)
    with open("output/d-words/freq-" + date + ".tsv", "w") as file:
        for k, v in fdist_tokens.most_common():
            file.write(f"{k}\t{v}\n")
    bgs = bigrams(tokens)
    fdist_bigram = FreqDist(bgs)
    with open("output/d-2gram/bigram-" + date + ".tsv", "w") as file:
        for k, v in fdist_bigram.most_common():
            file.write(f"{k}\t{v}\n")


def handle_sentence(sent, trie, parsed_sentences):
    left = 0
    right = 0
    parsed = []
    while right < len(sent):
        word = sent[left : right + 1]
        if word not in missing:
            while right < len(sent) and (
                (sent[left : right + 1].isalnum() and sent[left : right + 1].isascii())
                or sent[left : right + 1] in trie
                or trie.keys(sent[left : right + 1])
            ):
                right += 1
        if left == right:
            right += 1
            missing[sent[left:right]] += 1
            word = sent[left]
        else:
            word = sent[left:right]
            found[word] += 1
        parsed += [word]
        left += len(word)
        right = left
    parsed_sentences += [parsed]


def parse(date, trie, corpus):
    parsed_sentences = []
    for sent in corpus:
        handle_sentence(sent, trie, parsed_sentences)
    update_fdist(date, parsed_sentences)


def construct_trie(mode):
    words = import_data(mode)
    trie = marisa_trie.Trie(words)
    log(f"Trie created, size: {len(trie)}")
    return trie


def process(mode, trie):
    log("Extracting data to form corpus.")
    if mode == "test":
        corpus = [
            "𨋢字唔喺入面",
            "我今日食魚",
            "佢去食早餐",
            "我唔食魚，我食飯",
            "早餐食魚唔食飯",
            "而家食午餐",
            "你去食飯",
            "我唔食魚生飯",
            "我而家食飯",
            "我哋而家去食早餐",
            "你去唔去食飯",
            "食飯唔食",
            "我哋好早食飯",
            "早餐食飯，午餐食魚生",
            "飯我唔食",
            "早餐我唔食飯",
        ]
        parse("testing", trie, corpus)

    elif mode == "real":
        all_dates = [d for d in glob.iglob(mypath + "*")]
        total_dates = len(all_dates)
        log(f"Obtained File list: {total_dates} dates in total.")

        for i, datepath in enumerate(all_dates):
            all_articles = [f for f in glob.iglob(datepath + "/*/index.html")]
            log(
                f"{i+1:6}/{total_dates}: Processing {len(all_articles)} articles from {datepath}"
            )
            corpus = []
            for f in all_articles:
                corpus += [getcontent(f)]
            parse(datepath[-8:], trie, corpus)


def print_lists():
    log("Printing Missing Words")
    with open("output/missing.tsv", "w") as file:
        for k, v in missing.most_common():
            file.write(f"{k}\t{v}\n")

    log("Printing Found Words")
    with open("output/found.tsv", "w") as file:
        for k, v in found.most_common():
            file.write(f"{k}\t{v}\n")


# Run!
log(f"Start!\nMODE:{mode}")
process(mode=mode, trie=construct_trie(mode))
print_lists()
log("Done!")