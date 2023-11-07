import re
import json
import nltk
import time
import torch
from torch.utils.data import Dataset
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.tokenize import TweetTokenizer, casual
t_tokenizer = TweetTokenizer()

nltk.download('stopwords') # If needed
nltk.download('punkt') # If needed
nltk.download('wordnet') # If needed
nltk.download('averaged_perceptron_tagger')# If needed

stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords = stopwords.union(["--", "``", "''", "'", "(", ")", "|", "-", "`", "'", "..", ":", "&", "\"", "\/", "\\"])
# tweet pos tagset : http://www.cs.cmu.edu/~ark/TweetNLP/gimpel+etal.acl11.pdf
POS_TAGSET = ["N", "O", "S", "^", "Z", "L", "M", "V", "A", "R", "!", "D", "P", "&", "T", "X", "Y", "#", "@", "~", "U",
              "E", "$", ",", "G", "EJ"]


# For the LSTM model, create a glossary
def make_tokenizer(src_text, min_count=2):
    word_count = {}

    for texts in src_text:
        for w in texts:
            word_count[w] = word_count.get(w, 0) + 1

    print("original corpus contains", len(word_count), "words")

    vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2, "<sep>": 3}
    idx = 4
    for k, v in word_count.items():
        if k != "<sep>" and v > min_count:
            vocab[k] = idx
            idx += 1
    print("final vocab contains", idx, "words")

    return vocab


# Convert text sequence to word id
def tokenize(text, vocab):
    res = []
    unk = vocab["<unk>"]
    for w in text:
        res.append(vocab.get(w, unk))
    return res

# bert tokenizer
def bert_tokenize(text, tokenizer):
    tokens = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return input_ids


def get_tweet_token(string):
    return [wordnet_lemmatizer.lemmatize(token)
            for token in t_tokenizer.tokenize(string)
            if not re.search(casual.URLS, token)
            and token not in stopwords
            and not token.startswith("http")
            ]

# loading dataset from local directory
def load_local_dataset(dir, encoding='utf-8-sig', isNum=False, isBert=False):
    src_text = []
    with open(dir, 'r', encoding=encoding, errors='ignore') as f:
        raw_dataset = f.readlines()
    for line in raw_dataset:
        if line != '\n' and line != '':
            line = line.replace('\n', '')
            if isNum:
                line = int(line)
            else:
                if isBert:
                    pass
                else:
                    line = get_tweet_token(line)
            src_text.append(line) 
    return src_text


class FakeNewsDataset(Dataset):
    def __init__(self, src_texts, tgt_labels, padding_index=0):
        super().__init__()
        self.src_texts = src_texts
        self.tgt_labels = tgt_labels
        self.padding_index = padding_index

    def __len__(self):
        return len(self.tgt_labels)

    def __getitem__(self, index):
        return {"texts": self.src_texts[index], "labels": self.tgt_labels[index]}

    def padding(self, text):
        # Pad text sequence to the same length
        lens = []
        for t in text:
            lens.append(len(t))
        max_len = max(lens)
        
        mask = []
        pad_text = []
        for i, t in zip(lens, text):
            pad_len = max_len - i
            mask.append([1] * i + [0] * pad_len)
            pad_text.append(t + [self.padding_index] * pad_len)
        
        return pad_text, mask

    def collate_fn(self, batch):
        # Process the data of the entire batch and convert the filled data to tensor of PyTorch
        batch_tensor = dict()
        texts = []
        labels = []
        
        for x in batch:
            texts.append(x["texts"])
            labels.append(x["labels"])
        
        pad_text, mask = self.padding(texts)
        
        batch_tensor["texts"] = torch.LongTensor(pad_text)
        batch_tensor["mask"] = torch.Tensor(mask)
        batch_tensor["labels"] = torch.LongTensor(labels)
        return batch_tensor