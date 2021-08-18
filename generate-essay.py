import pandas as pd

essay_cantonese = pd.read_csv(
    "source/essay-cantonese.txt",
    header=None,
    names=["char", "freq"],
    sep="\t",
)
found = pd.read_csv(
    "output/found.tsv",
    header=None,
    names=["char", "freq"],
    sep="\t",
)

file = open("output/essay-new.txt", "w")

# All words from essay_cantonese will be kept.
combined = (
    pd.merge(essay_cantonese, found[found.freq > 10], on=["char"], how="outer")
    .set_index(["char"])
    .sum(axis=1)
)
combined.to_csv("output/essay-new.txt", sep="\t", header=None, float_format="%u")
