import sys, os, getopt, time, glob, collections, re
import marisa_trie
from bs4 import BeautifulSoup
import lxml, cchardet
import opencc
from nltk import flatten, bigrams, ngrams, FreqDist
from nltk.lm.preprocessing import pad_both_ends
import pandas as pd

mode = "real"
mypath = "data/"
task = "freq"
punc = "[%:.,#/\\\-\u2000-\u206F\u3002\uff1f\uff01\uff0c\u3001\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09\u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u300c\u300d\ufe43\ufe44\u3014\u3015\u2026\u2014\uff5e\ufe4f\uffe5\"']"
mingram = 2
maxgram = 6


opts, args = getopt.getopt(
    sys.argv[1:], "trhp:g:", ["test", "real", "head", "path=", "get="]
)
for o, a in opts:
    if o in ("-t", "--test"):
        mode = "test"
    if o in ("-r", "--real"):
        mode = "real"
    if o in ("-h", "--head"):
        mode = "head"
    if o in ("-p", "--path"):
        mypath = a
    if o in ("-g", "--get"):
        task = a

missing = collections.Counter()
found = collections.Counter()
all_bigrams = FreqDist()
swc_stat = []

## Prepare directories
os.system("mkdir -p output; cd output; mkdir -p d-2gram d-words d-ngram")
logfile = open("output/getfreq.log", "a")

## Prepare Cleanup Text
cleanup = []
with open('cleanup.txt', 'r') as file:
    cleanup = file.readlines()
cleanup = [l.strip() for l in sorted(cleanup, key=len, reverse=True)]

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

    elif mode == "real" or mode == "head":
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
        log(f"Added {len(words_t2hk)} Hong Kong orthographic vairants.")

        # Limit word size to four characters
        trimmed = list(filter(lambda w: len(str(w)) < 5, words + words_t2hk))
        log(f"Trimmed from {len(words + words_t2hk)} to {len(trimmed)}.")
        return trimmed


def getcontent(f):
    with open(f, "r") as file:
        s = file.read()
        soup = BeautifulSoup(s, "lxml")
        if (soup.title):
            soup.title.decompose()
        h1tag = soup.h1
        title = ''
        if (h1tag):
            title = soup.h1.get_text().strip().replace('\n',' ').replace('\r',' ')
            soup.h1.decompose()
        content = soup.get_text(' ')
        for line in cleanup:
            content = content.replace(line, '')
        return {'title': title.strip(), 'content': content.strip()}


def print_fdist(label, tokens):
    fdist_tokens = FreqDist(tokens)
    with open("output/" + label + ".tsv", "w") as file:
        for k, v in fdist_tokens.most_common():
            file.write(f"{k}\t{v}\n")


def update_fdist(date, parsed_sentences):
    tokens = flatten(parsed_sentences)
    print_fdist("d-words/freq-" + date, tokens)
    # print_fdist("d-2gram/bigram-" + date, bigrams(tokens))


def handle_sentence(sent, trie, parsed_sentences):
    left = 0
    right = 0
    parsed = []
    while right < len(sent):
        word = sent[left : right + 1]
        peep = word
        if word not in missing:
            while right < len(sent) and (
                (peep.isalnum() and peep.isascii()) or peep in trie or trie.keys(peep)
            ):
                if peep in trie:
                    # log(f"Updating word to {peep}.")
                    word = peep
                elif peep.isalnum() and peep.isascii():
                    # log(f"Updating Alphanumeric word {word}.")
                    word = peep
                else:
                    # log(f"{peep} not in trie and not alphanum, longest word is {word}.")
                    pass
                right += 1
                peep = sent[left : right + 1]
            if not (word.isalnum() and word.isascii()):
                found[word] += 1
        else:
            right += 1
            missing[word] += 1
        parsed += [word]
        left += len(word)
        right = left
    parsed_sentences += [parsed]


def parse(date, trie, corpus):
    if task == "freq":
        get_freq(date, trie, [obj["content"] for obj in corpus])
    elif task == "grams":
        get_grams(date, trie, [obj["content"] for obj in corpus])
    elif task == "swc":
        get_swc_stat(date, trie, corpus)


def get_freq(date, trie, corpus):
    parsed_sentences = []
    for sent in corpus:
        segmented = re.sub(punc, " ", sent).split()
        for seg in segmented:
            handle_sentence(seg, trie, parsed_sentences)
    update_fdist(date, parsed_sentences)


def get_grams(date, trie, corpus):
    seg_gram = {}
    for i in range(mingram, maxgram + 1):
        seg_gram[i] = []
    for sent in corpus:
        segmented = re.sub(punc, " ", sent).split()
        for seg in segmented:
            for i in range(mingram, maxgram + 1):
                seg_gram[i] += re.findall("(?=(\S{" + str(i) + "}))", seg)
    for i in range(mingram, maxgram + 1):
        print_fdist(f"d-ngram/char{i}gram-{date}", seg_gram[i])


canto_unique = "[嘅嗰啲咗佢喺咁睇冇啩唔哋𠵱畀俾嚟]"
mando_feature = "[那是的些沒了不他]"
mando_exclude = "(剎那|亞利桑那|關塔那摩|巴塞羅那|那不勒斯|斯堪地那維亞|圭亞那|熱那亞|毗盧遮那|支那|是次|是日|是非|利是|唯命是從|頭頭是道|誰是誰非|似是而非|自以為是|俯捨皆是|撩是鬥非|莫衷一是|大吉利是|尤其是|目的|紅的|綠的|的士|波羅的海|的確|些微|些小|淹沒|沉沒|沒收|湮沒|口沒遮攔|沒落|埋沒|沒頂|了解|沒完沒了|了結|未了緣|了無生趣|不了了之|了哥|直截了當|了斷|一目了然|了無牽掛|了無新意|了然於胸|不過|不滿|不適|不如|不妨|不俗|不宜|不僅|不必|不利|不符|不果|不然|不死|急不及待|不一|意想不到|不論|不約而同|不忠|永不|不入|樂此不疲|有所不同|不足|不值|不妙|滔滔不絕|不務|不外乎|不良|不知幾|不斷|不同|不得了|他人|他信|他加祿|他國|他山之石|他日|他殺|他處|他鄉|其他|利他|無他|排他|左右而言他|維他|馬爾他|馬耳他)"
canto_feature = "[嗰係嘅啲冇咗唔佢]"
canto_exclude = "(關係|吱唔|咿唔)"
allquotes = "「[^「」]*」"


def count_canto_feature(s):
    return len(re.findall(canto_feature, s)) - len(re.findall(canto_exclude, s))


def count_mando_feature(s):
    return len(re.findall(mando_feature, s)) - len(re.findall(mando_exclude, s))


def get_swc_stat(date, trie, corpus):
    for obj in corpus:
        sent = obj["content"]
        quote = "".join(re.findall(allquotes, sent))
        matrix = re.sub(allquotes, " ", sent)
        quote_canto_unique = len(re.findall(canto_unique, quote))
        matrix_canto_unique = len(re.findall(canto_unique, matrix))
        quote_mando_feature = count_mando_feature(quote)
        quote_canto_feature = count_canto_feature(quote)
        matrix_mando_feature = count_mando_feature(matrix)
        matrix_canto_feature = count_canto_feature(matrix)
        swc_stat.append(
            {
                "date": date,
                "title": obj["title"],
                "path": obj["path"],
                "totallength": len(sent),
                "quotelength": len(quote),
                "matrixlength": len(matrix),
                "quote_canto_unique": quote_canto_unique,
                "matrix_canto_unique": matrix_canto_unique,
                "quote_mando_feature": quote_mando_feature,
                "quote_canto_feature": quote_canto_feature,
                "matrix_mando_feature": matrix_mando_feature,
                "matrix_canto_feature": matrix_canto_feature,
            }
        )


def construct_trie(mode):
    words = import_data(mode)
    trie = marisa_trie.Trie(words)
    log(f"Trie created, size: {len(trie)}")
    return trie


def process(mode, trie):
    log("Extracting data to form corpus.")
    if mode == "test":
        content = [
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
        parse(
            "testing",
            trie,
            [{"path": i, "title": "", "content": content} for i, content in enumerate(content)],
        )

    elif mode == "real" or mode == "head":
        all_dates = [d for d in glob.iglob(mypath + "*")]
        if mode == "head":
            all_dates = all_dates[:20]
        total_dates = len(all_dates)
        log(f"Obtained File list: {total_dates} dates in total.")

        for i, datepath in enumerate(all_dates):
            all_articles = [f for f in glob.iglob(datepath + "/*/index.html")]
            log(
                f"{i+1:6}/{total_dates}: Processing {len(all_articles)} articles from {datepath}"
            )
            corpus = []
            for f in all_articles:
                corpus += [{"path": f, **getcontent(f)}]
            parse(datepath[-8:], trie, corpus)


def print_lists():

    if task == "swc":
        df = pd.DataFrame(swc_stat)
        df.to_csv("output/swc_stat.tsv", sep="\t")
    elif task == "freq":
        log("Printing Missing Words")
        with open("output/missing.tsv", "w") as file:
            for k, v in missing.most_common():
                file.write(f"{k}\t{v}\n")

        log("Printing Found Words")
        with open("output/found.tsv", "w") as file:
            for k, v in found.most_common():
                file.write(f"{k}\t{v}\n")
    elif task == "grams":
        pass


# Run!
log(f"Start!\nMODE:{mode}")
process(mode=mode, trie=construct_trie(mode))
print_lists()
log("Done!")
