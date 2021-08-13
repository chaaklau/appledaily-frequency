import glob
from nltk import FreqDist
import pandas as pd

mingram = 2
maxgram = 6
for i in reversed(range(mingram, maxgram + 1)):
    files = [d for d in glob.iglob(f"output/d-ngram/char{i}gram*")]
    fdist = FreqDist()
    for f in files:
        print(f)
        df = pd.read_csv(
            f,
            header=None,
            names=["token", "freq"],
            sep="\t",
            quoting=3,
            index_col="token",
        )
        fdist += FreqDist(df.to_dict("dict")["freq"])

    with open(f"output/combined-{i}gram.tsv", "w") as file:
        for k, v in fdist.most_common():
            if v > 10:
                file.write(f"{k}\t{v}\n")
