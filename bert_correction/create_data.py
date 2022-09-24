import json
from utils import try_create_dir
import re
from copy import deepcopy
from random import seed, choice


def create_data_with_perturb(essay_file_name, comment_file_name, out_file_name):
    seed(2021)
    essays = []
    comments = []

    all_words = []

    with open(essay_file_name, encoding='utf-8') as essay_f:
        with open(comment_file_name, encoding='utf-8') as comment_f:
            for essay, comment in zip(essay_f, comment_f):
                essay = essay.strip()
                comment = comment.strip()
                essays.append(essay)
                comments.append(comment)
                words = comment.split('<eod>')[1].split('<eop>')
                all_words.extend(words)

    unique_words = []
    record = set()
    for word in all_words:
        if word in record:
            continue
        record.add(word)
        unique_words.append(word)

    res = []

    for essay, comment in zip(essays, comments):
        api_info, essay = essay.split('<eod>', maxsplit=1)
        api_keywords = re.split(r'：|；|。', api_info)
        api_keywords = [word for word in api_keywords if word]
        mode, comment_keywords = comment.split('<eod>')
        comment_keywords = comment_keywords.split('<eop>')
        true_keywords = deepcopy(comment_keywords)
        if mode == '0':
            # 优点才考虑api的keywords
            for word in api_keywords:
                if word not in true_keywords:
                    true_keywords.append(word)

        false_keywords = []
        for i in range(len(true_keywords)):
            word = choice(unique_words)
            while word in true_keywords:
                word = choice(unique_words)
            false_keywords.append(word)

        for word in true_keywords:
            res.append({
                'essay': essay,
                'word': mode + '<eod>' + word,
                'label': 1
            })

        for word in false_keywords:
            res.append({
                'essay': essay,
                'word': mode + '<eod>' + word,
                'label': 0
            })

    print(f"{out_file_name} len:{len(res)}")

    with open(out_file_name, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    try_create_dir('data')
    for split in ['test', 'val', 'train']:
        create_data_with_perturb(f'../t5/data/{split}_clean.source', f'../t5/data/{split}_notitledup_tfidfkws.target',
                                 f'./data/{split}.json')
