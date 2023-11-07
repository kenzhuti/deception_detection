import torch
import random
import json
from src.data_lstm import load_local_dataset, make_tokenizer, bert_tokenize, FakeNewsDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support
import os
from src.attack_train import FGM
from src.ema import EMA
from src.model_bert import BertFakeNewsDetect
from src.loss import binary_cross_entropy, FocalLoss, LabelSmoothingCrossEntropy
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='./datasets/fakeNews_/train_x.csv')
parser.add_argument('--train_labels', type=str, default='./datasets/fakeNews_/train_y.csv')
parser.add_argument('--dev_path', type=str, default='./datasets/fakeNews_/dev_x.csv')
parser.add_argument('--dev_labels', type=str, default='./datasets/fakeNews_/dev_y.csv')
parser.add_argument('--test_path', type=str, default='./datasets/fakeNews_/test_x.csv')
parser.add_argument('--test_labels', type=str, default='./datasets/fakeNews_/test_y.csv')

parser.add_argument('--bert_path', type=str, default='./datasets/bert-base-uncased-en/')
parser.add_argument('--save_path', type=str, default='./save/')
parser.add_argument('--model_name', type=str, default='all', choices=['all', 'no_drop_out', 'no_attention'])
parser.add_argument('--n_epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--embed_dim', type=int, default=768)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--embed_dropout', type=float, default=0.5)
parser.add_argument('--lstm_dropout', type=float, default=0.2)
parser.add_argument('--fc_dropout', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--nums_layer', type=int, default=2)


parser.add_argument('--fgm', action="store_true")
parser.add_argument('--ema', action="store_true")
parser.add_argument('--aug', action="store_true")
parser.add_argument('--loss_type', type=str, default="lce", choices=["fl", "lce", "bce"])



# args = parser.parse_args()
args = parser.parse_args()

torch.manual_seed(888)
torch.cuda.manual_seed(888)
random.seed(888)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def make_data():
    train_text = load_local_dataset(args.train_path, isBert=True)
    train_label = load_local_dataset(args.train_labels, isNum=True)
    test_text = load_local_dataset(args.test_path, isBert=True)
    test_label = load_local_dataset(args.test_labels, isNum=True)

    print("Num of training items is", len(train_label))
    print("Ratio of positive samples", sum(train_label) / float(len(train_label)))
    print("Num of test items is", len(test_label))
    print("Ratio of positive samples", sum(test_label) / float(len(test_label)))

    print("------------------------------------------------")

    if args.aug:
        aug_text = []
        aug_label = []
        split_len = 2
        for text, label in zip(train_text, train_label):
            tokens = text.split(" ")
            mid_tokens = tokens[len(tokens) // 2: len(tokens) // 2 + split_len]
            aug_tokens = tokens[:len(tokens) // 2] + tokens[len(tokens) // 2 + split_len:] + mid_tokens
            aug_text.append(" ".join(aug_tokens))
            aug_label.append(label)

        train_text.extend(aug_text)
        train_label.extend(aug_label)
    
    # Import bert's tokenizer
    tokenizer = BertTokenizer(os.path.join(args.bert_path, 'vocab.txt'))
    
    # Use bert's tokenizer to segment the text
    train_text_vec = []
    for i in train_text:
        train_text_vec.append(bert_tokenize(i, tokenizer))
    
    # Create a training data set, follow the PyTorch method to create
    TrainDataset = FakeNewsDataset(train_text_vec, train_label)

    total_sample = 2 * (len(train_label) - sum(train_label))
    sweight = [2 if i else 1 for i in train_label]
    
    # Create a sampler of the data set, and generate a batch of data according to the sampling method of the sampler
    from torch.utils.data.sampler import WeightedRandomSampler
    wsampler = WeightedRandomSampler(sweight, num_samples=total_sample, replacement=True)

    TrainDataloader = DataLoader(TrainDataset, batch_size=args.batch_size, collate_fn=TrainDataset.collate_fn,
                                 num_workers=4, sampler=wsampler)

    test_text_vec = []
    for i in test_text:
        test_text_vec.append(bert_tokenize(i, tokenizer))

    TestDataset = FakeNewsDataset(test_text_vec, test_label)
    TestDataloader = DataLoader(TestDataset, batch_size=args.batch_size, collate_fn=TestDataset.collate_fn,
                                shuffle=False,
                                num_workers=4, sampler=None)

    torch.save(TrainDataloader, "TrainDataloader.pth")
    torch.save(TestDataloader, "TestDataloader.pth")
    return TrainDataloader, TestDataloader


def train(train_loader, test_loader):
    model = BertFakeNewsDetect(args)   # Create fake news detection model
    model = model.to(device)

    if args.fgm:
        fgm = FGM(model)

    if args.ema:
        ema = EMA(model, 0.99)
        ema.register()

    optimizer = AdamW(model.parameters(), lr=args.lr)  # Create AdamW optimizer
    
    best_f1 = 0
    early_stopping_count = 0

    loss_weight = torch.FloatTensor([1.0, 2.0])
    loss_weight = loss_weight.to(device)

    if args.loss_type == "fl":
        loss_fn = FocalLoss()
    elif args.loss_type == "bce":
        loss_fn = binary_cross_entropy
    else:
        loss_fn = LabelSmoothingCrossEntropy()

    for epoch_id in range(args.n_epochs):
        model.train()
        total_loss = 0.0
        for x in tqdm(train_loader):
            optimizer.zero_grad()
            text = x["texts"].to(device)
            mask = x["mask"].to(device)
            label = x["labels"].to(device)

            p = model(text, mask)

            loss = loss_fn(p, label)  # Binary cross entropy loss function
            loss.backward()

            if args.fgm:
                fgm.attack()
                p_adv = model(text, mask)
                loss_adv = loss_fn(p_adv, label)
                loss_adv.backward()
                fgm.restore()

            optimizer.step()
            if args.ema:
                ema.update()

            total_loss += loss.data.item()

        print('--------------------Epoch: {}----------------------'.format(epoch_id))
        print('train loss: {}'.format(total_loss / len(train_loader)))

        if args.ema:
            ema.apply_shadow()
        model.eval()
        preds = []
        tgt_labels = []
        loss_total = []
        for x in test_loader:
            text = x["texts"].to(device)
            mask = x["mask"].to(device)
            label = x["labels"].to(device)

            p = model(text, mask)

            top_n, top_i = p.topk(1)  # top_n is the corresponding element value, top_i is the corresponding element id, which is the predicted label 0 or 1
            preds += top_i.squeeze(dim=1).tolist()

            loss = binary_cross_entropy(p, label, loss_weight)
            loss_total.append(loss.item())
            # print("x[labels]=", x["labels"], type(x["labels"]))
            tgt_labels += x["labels"].tolist()

        if args.ema:
            ema.restore()

        test_loss = sum(loss_total) / len(loss_total)
        print("test loss : ", test_loss)

        p, r, f, _ = precision_recall_fscore_support(tgt_labels, preds, pos_label=1, average="binary")

        cnt = 0
        for pred, tgt in zip(preds, tgt_labels):
            if pred == tgt:
                cnt += 1

        print("Performance:")
        print("Precision = {} Recall = {} F1 = {}".format(p, r, f))
        print("ACC = {}".format(cnt * 1.0 / len(preds)))
        print("--------------------------------------------------------------------")
        if f > best_f1:
            best_f1 = f
            early_stopping_count = 0
            if args.ema:
                ema.apply_shadow()
            torch.save(model.state_dict(), args.save_path + 'model_' + args.model_name + '.pt')
            if args.ema:
                ema.restore()
        else:
            if early_stopping_count > 5:
                break
            early_stopping_count += 1


def test(dev_loader):
    model = BertFakeNewsDetect(args)
    model.load_state_dict(torch.load(args.save_path + 'model_' + args.model_name + '.pt'))
    model = model.to(device)
    model.eval()

    preds = []
    tgt_labels = []
    for x in dev_loader:
        text = x["texts"].to(device)
        mask = x["mask"].to(device)

        p = model(text, mask)
        p = torch.softmax(p, 1)
        y = p[:, 1] - p[:, 0] # The probability of a label of 1 (predicted as fake news) minus the probability of a label of 0 (predicted as true news)
        y = y > 0  # If the probability of fake news is higher than the probability of real news, it means that the prediction is fake news, otherwise it is real news
        preds += y.int().tolist()

        tgt_labels += x["labels"].tolist()
    p, r, f, _ = precision_recall_fscore_support(tgt_labels, preds, pos_label=1, average="binary")
    cnt = 0
    for pred, tgt in zip(preds, tgt_labels):
        if pred == tgt:
            cnt += 1

    print("Performance:")
    print("Precision = {} Recall = {} F1 = {}".format(p, r, f))
    print("ACC = {}".format(cnt * 1.0 / len(preds)))
    print("--------------------------------------------------------------------")


if __name__ == '__main__':
    train_loader, test_loader = make_data()
    # train_loader = torch.load("TrainDataloader.pth")
    # test_loader = torch.load("TestDataloader.pth")
    train(train_loader, test_loader)
    print('Train over!!!')
    test(test_loader)
    # print_test(test_loader)
