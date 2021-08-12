# Frequency Table for Hong Kong Chinese (with Cantonese) from Apple Daily

This is a frequency list generated from Apple Daily plain text data, obtained from this backup source: https://github.com/appledailybackup/appledaily-archive-directory. All files from `apple-articles-plaintext-20020101-20210620.zip` have been used.

Although Apple Daily is no longer operating in Hong Kong, the license status of the orginal dataset is unknown. This repo does not contain any source file, nor is the content from the source reconstructable from the data provided here. The author believes that using the frequency data from this repository does not violate the copyrights of Apple Daily.

## Construction of the frequence list

The original data was word segmented using a trie (by longest string matching). These word lists have been used in the process:
- [Rime Essay(八股文)](https://github.com/rime/rime-essay)
- [Rime Cantonese](https://github.com/rime/rime-cantonese)
- [The words.hk list](https://words.hk/faiman/analysis/existingwordcount/)

Content from Rime projects have been converted to Hong Kong variants with the [OpenCC](https://github.com/BYVoid/OpenCC) tool before segmentation.

## Generating the frequency list and bigrams

`getFreq.py` generates frequency and bigram from the source files.

*Real mode is used by default if the following flags are not configured*
- `-t` or `--test` enables the testing mode, which generates bigram
- `-r` or `--real` enables the real mode, which will 

*The default path is `data/`*
- `-p` or `--path` specifies the root of the dataset. 

E.g.

```
python3 getFreq.py -r -p /User/ABC/AppleDaily/Data/
```

These files will be generated under `output`:
- `found.tsv` contains a frequency list of all words, 
- `missing.tsv` contains all missing characters, 
- The `d-2gram` folder stores bigram data sorted by date.
- The `d-words` folder stores word frequency table sorted by date.

## Release

You can use the frequency data without obtaining the source files. The `release/` folder of this repository contains a `.txt` file containing headwords and frequency. All numerals have been removed. All words that contain only letters with a word count below 50 have also been removed.

## Acknowledgement

I would like to thank everyone from [CanCLID](https://github.com/CanCLID).

## Reference

To be added.
