import torch
import random
import json
from src.data_lstm import load_local_dataset, make_tokenizer, tokenize, FakeNewsDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from src.attack_train import FGM
from src.ema import EMA
from src.model_lstm import FakeNewsDetect
from src.loss import binary_cross_entropy, FocalLoss, LabelSmoothingCrossEntropy
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='./datasets/fakeNews_/train_x.csv')
parser.add_argument('--train_labels', type=str, default='./datasets/fakeNews_/train_y.csv')
parser.add_argument('--dev_path', type=str, default='./datasets/fakeNews_/dev_x.csv')
parser.add_argument('--dev_labels', type=str, default='./datasets/fakeNews_/dev_y.csv')
parser.add_argument('--test_path', type=str, default='./datasets/fakeNews_/test_x.csv')
parser.add_argument('--test_labels', type=str, default='./datasets/fakeNews_/test_y.csv')

parser.add_argument('--save_path', type=str, default='./save/')
parser.add_argument('--model_name', type=str, default='all', choices=['all', 'no_drop_out', 'no_attention'])
parser.add_argument('--n_epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--embed_dim', type=int, default=300)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--embed_dropout', type=float, default=0.5)
parser.add_argument('--lstm_dropout', type=float, default=0.2)
parser.add_argument('--fc_dropout', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--nums_layer', type=int, default=2)

parser.add_argument('--fgm', action="store_true")
parser.add_argument('--ema', action="store_true")
parser.add_argument('--aug', action="store_true")
parser.add_argument('--loss_type', type=str, default="lce", choices=["fl", "lce", "bce"])

args = parser.parse_args()

torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def make_data():
    train_text = load_local_dataset(args.train_path)
    train_label = load_local_dataset(args.train_labels, isNum=True)
    test_text = load_local_dataset(args.test_path)
    test_label = load_local_dataset(args.test_labels, isNum=True)
  
    print("Num of training items is", len(train_label))
    print("Ratio of positive samples", sum(train_label) / float(len(train_label)))
    print("Num of test items is", len(test_label))
    print("Ratio of positive samples", sum(test_label) / float(len(test_label)))

    print("------------------------------------------------")

    # Data augmentation, achieved by swapping two characters in the middle and at the end of a sentence
    if args.aug:
        print("aug set")
        aug_text = []
        aug_label = []
        split_len = 2
        for tokens, label in zip(train_text, train_label):
            # tokens = text.split(" ")
            mid_tokens = tokens[len(tokens) // 2: len(tokens) // 2 + split_len]
            aug_tokens = tokens[:len(tokens) // 2] + tokens[len(tokens) // 2 + split_len:] + mid_tokens
            aug_text.append(aug_tokens)
            aug_label.append(label)

        train_text.extend(aug_text)
        train_label.extend(aug_label)

    vocab = make_tokenizer(train_text, min_count=0)
    f = open("./save/vocab.json", "w")
    json.dump(vocab, f)
    args.vocab_size = len(vocab)

    train_text_vec = []
    for i in train_text:
        train_text_vec.append(tokenize(i, vocab))

    TrainDataset = FakeNewsDataset(train_text_vec, train_label)

    total_sample = 2 * (len(train_label) - sum(train_label))
    sweight = [2 if i else 1 for i in train_label]

    from torch.utils.data.sampler import WeightedRandomSampler
    wsampler = WeightedRandomSampler(sweight, num_samples=total_sample, replacement=True)

    TrainDataloader = DataLoader(TrainDataset, batch_size=args.batch_size, collate_fn=TrainDataset.collate_fn,
                                 num_workers=4, sampler=wsampler)

    test_text_vec = []
    for i in test_text:
        test_text_vec.append(tokenize(i, vocab))

    TestDataset = FakeNewsDataset(test_text_vec, test_label)
    TestDataloader = DataLoader(TestDataset, batch_size=args.batch_size, collate_fn=TestDataset.collate_fn,
                                shuffle=False,
                                num_workers=4, sampler=None)

    return TrainDataloader,  TestDataloader


def train(train_loader, test_loader):
    model = FakeNewsDetect(args)
    model = model.to(device)

    if args.fgm:
        print("fgm set")
        fgm = FGM(model)

    if args.ema:
        print("ema set")
        ema = EMA(model, 0.999)
        ema.register()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

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
            loss = loss_fn(p, label)
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
        scheduler.step()
        print('--------------------Epoch: {}----------------------'.format(epoch_id))
        print('train loss: {}'.format(total_loss / len(train_loader)))

        # Apply EMA and update the model parameters
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

            loss = loss_fn(p, label)
            loss_total.append(loss.item())
            # print("x[labels]=", x["labels"], type(x["labels"]))
            tgt_labels += x["labels"].tolist()

        # After evaluation, restore the original model parameters
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
    model = FakeNewsDetect(args)
    model.load_state_dict(torch.load(args.save_path+'model_'+args.model_name+'.pt'))
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
    train_loader,  test_loader = make_data()
    train(train_loader, test_loader)
    print('Train over!!!')
    test(test_loader)
    # print_test(test_loader)
