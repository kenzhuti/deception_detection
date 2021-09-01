import os
import pandas as pd

root_path = r'../datasets/'
# path of fake news dataset
fn_path = root_path + 'fakeNewsDataset/'
# path of celebrity dataset
c_path = root_path + 'celebrityDataset/'
fn_true_path = fn_path + 'legit/'
fn_false_path = fn_path + 'fake/'
c_true_path = c_path + 'legit/'
c_false_path = c_path + 'fake/'


def concatenate2CSV(path, name: str, theme, label):
    files = os.listdir(path)
    title_list = []
    data = pd.DataFrame()
    content_list = []
    for file in files:
        position = path + file
        with open(position, 'r') as f:
            lines = f.readlines()
            # lines.remove('\n')
            list(map(str.strip, lines))
            title = lines[0]
            content = [line for line in lines[1:] if line.strip()]
            title_list.append(title)
            for l in content:
                content_list.append(l)
    data['title'] = pd.Series(title_list)
    data['content'] = pd.Series(content_list)
    data['theme'] = theme
    data['label'] = 0
    data.to_csv(root_path + name, header=True, index=False, encoding='utf_8_sig')


concatenate2CSV(fn_true_path, "news_true.csv", theme="news", label=0)
concatenate2CSV(fn_false_path, "news_false.csv", theme="news", label=1)
concatenate2CSV(c_true_path, "celebrity_true.csv", theme="celebrity", label=0)
concatenate2CSV(c_false_path, "celebrity_false.csv", theme="celebrity", label=1)
