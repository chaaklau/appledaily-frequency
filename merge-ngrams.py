import glob
from nltk import FreqDist
import pandas as pd
import time
import marisa_trie
import os

logfile = open("output/getfreq.log", "a")
os.system("mkdir -p tmp")


def log(msg):
    print(msg)
    logfile.write(f'{time.strftime("%H:%M:%S", time.localtime())}\t{msg}\n')
    logfile.flush()


bigrams = pd.read_csv(
    "release/bigram.tsv",
    header=None,
    names=["char", "freq"],
    quoting=3,
    sep="\t",
)

trie = marisa_trie.Trie(bigrams.char.to_list())

mingram = 2
maxgram = 6


def writeDist(label, dist, threshold):
    with open(label, "w") as file:
        for k, v in dist.most_common():
            if v > threshold:
                file.write(f"{k}\t{v}\n")


for i in reversed(range(mingram, maxgram + 1)):

    # First pass
    files = [d for d in glob.iglob(f"output/d-ngram/char{i}gram*")]
    fdist = FreqDist()
    for j, f in enumerate(files):
        log(f"{j+1:4}/{len(files)}: {f}")
        df = pd.read_csv(
            f,
            header=None,
            names=["token", "freq"],
            sep="\t",
            quoting=3,
            index_col="token",
        )
        fdist += FreqDist(
            # Aggregate all frequencies
            {
                k: v
                for (k, v) in df.to_dict("dict")["freq"].items()
                if str(k)[:2] in trie
            }
            # Count number of days each word is found
            # {k: 1 for (k, v) in df.to_dict("dict")["freq"].items() if str(k)[:2] in trie}
        )
        if (j + 1) % 50 == 0:
            writeDist(f"tmp/combined-temp-{j+1}-{i}gram.tsv", fdist, 2)
            fdist = FreqDist()
    # Remaining
    writeDist(f"tmp/combined-temp-last-{i}gram.tsv", fdist, 2)
    fdist = FreqDist()

    # Second pass
    files = [d for d in glob.iglob(f"tmp/combined-temp-*-{i}gram*")]
    fdist = FreqDist()
    for j, f in enumerate(files):
        log(f"{j+1:4}/{len(files)}: {f}")
        df = pd.read_csv(
            f,
            header=None,
            names=["token", "freq"],
            sep="\t",
            quoting=3,
            index_col="token",
        )
        fdist += FreqDist(df.to_dict("dict")["freq"])
    writeDist(f"output/combined-{i}gram.tsv", fdist, 10)
