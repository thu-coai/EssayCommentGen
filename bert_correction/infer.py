import json

from transformers.utils.dummy_pt_objects import LED_PRETRAINED_MODEL_ARCHIVE_LIST
from main import ModelAPI
import torch
from tqdm import tqdm
import numpy as np


def infer(essay_file_name, kws_file_name, out_file_name):
    with open(essay_file_name, encoding='utf-8') as essay_f:
        with open(kws_file_name, encoding='utf-8') as kws_f:
            essay_lines = [line.strip() for line in essay_f]
            kws_lines = [line.strip() for line in kws_f]

    api = ModelAPI(device=torch.device('cuda'))

    avg_len = 0
    avg_len_new = 0

    print(f"sample num:{len(essay_lines)}")

    new_kws_lines = []
    limit = 0.15
    for essay, kws in tqdm(zip(essay_lines, kws_lines)):
        mode, words = kws.split('<eod>')[:2]
        words = words.split('<eop>')
        sents = [[essay, mode + '<eod>' + word] for word in words]
        scores = api.get_score(sents)  # ndarray
        cnt = (scores > limit).sum()
        # print(f"{cnt}/{len(scores)}")
        avg_len += len(scores)
        new_kws = []
        # idx = np.argmin(scores)
        # if scores[idx] < limit:
        #     new_kws = words[:idx] + words[idx+1:]
        # else:
        #     new_kws = words
        for score, word in zip(scores, words):
            if score > limit:
                new_kws.append(word)
        avg_len_new += len(new_kws)
        # if len(new_kws) != len(words):
        #     idx1 = essay.find('<extra_id_0>')
        #     idx2 = essay.find('<extra_id_1>')
        #     segment = essay[idx1:idx2].replace('<extra_id_0>','')
        #     print(f"segment:{segment}\nwords:{words}\nnew_words:{new_kws}")


        new_kws_lines.append(f"{mode}<eod>{'<eop>'.join(new_kws)}")

    print(
        f"old avg len:{avg_len / len(essay_lines)}, new avg len:{avg_len_new / len(essay_lines)}")

    with open(out_file_name, 'w', encoding='utf-8') as f:
        for line in new_kws_lines:
            f.write(line + '\n')


if __name__ == '__main__':
    infer('../t5/data/test_notitledup.source', '../t5/result/_cg_base_notitledup_tfidfkws_checkpoint-1695_test_notitledup_all.txt',
          '../t5/data/test_notitledup_generated_tfidfkws_corrected')
    