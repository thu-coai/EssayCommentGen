import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer, AdamW, HfArgumentParser, AutoConfig
from typing import Optional
from dataclasses import dataclass, field
from utils import get_logger
from metrics import MyAccuracy
import os
import json

logger = get_logger(__name__)


class Model(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)


class PLDataset(Dataset):
    def __init__(self, path):
        with open(path, encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class PLDataLoader(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_data_path = args.train_data_path
        self.val_data_path = args.val_data_path
        self.test_data_path = args.test_data_path
        self.train_batch_size = args.train_batch_size
        self.valtest_batch_size = args.valtest_batch_size
        self.mode = args.mode

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            logger.info('begin get data')

            self.train_dataset = PLDataset(self.train_data_path)
            self.val_dataset = PLDataset(self.val_data_path)
            logger.info('finish get data')

        if stage == 'test' or stage is None:
            logger.info('begin get data')
            self.test_dataset = PLDataset(self.test_data_path)
            logger.info('finish get data')

    def collate_fn(self, batch):
        # logger.debug(batch)
        srcs = [sample['essay'] for sample in batch]
        tgts = [sample['word'] for sample in batch]
        sents = [[src, tgt] for src, tgt in zip(srcs, tgts)]
        res = self.tokenizer(sents,
                             return_tensors='pt',
                             padding=True,
                             max_length=1024,
                             truncation=True)
        if self.mode != 'infer':
            labels = [sample['label'] for sample in batch]
            res['labels'] = torch.tensor(labels, dtype=torch.long)
        return res

    def train_dataloader(self, ):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          num_workers=32,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.valtest_batch_size,
                          num_workers=32,
                          pin_memory=True,
                          shuffle=False,
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.valtest_batch_size,
                          num_workers=32,
                          pin_memory=True,
                          shuffle=False,
                          collate_fn=self.collate_fn)


class PLModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path)
        self.config = AutoConfig.from_pretrained(args.pretrain_path)
        # self.config.max_position_embeddings = 1024
        # logger.debug(self.config)

        self.model = Model.from_pretrained(args.pretrain_path, config=self.config)
        logger.debug(self.model)
        logger.debug(self.model.bert.embeddings.position_embeddings)
        logger.info(f"tokenizer pad token:{self.tokenizer.pad_token}")
        old_weight = self.model.bert.embeddings.position_embeddings.weight
        new_rand_weight = torch.randn(512, self.config.hidden_size)
        new_weight = torch.cat([old_weight, new_rand_weight], dim=0)
        new_pos_emb = torch.nn.Embedding(1024, self.config.hidden_size, _weight=new_weight)
        self.model.bert.embeddings.position_embeddings = new_pos_emb
        self.model.bert.embeddings.register_buffer("position_ids", torch.arange(1024).expand((1, -1)))

        logger.debug(self.model.bert.embeddings.position_ids)

        self.tokenizer.add_special_tokens({
            # 'pad_token': '[PAD]',
            'additional_special_tokens': ['<eod>', '<eop>', '<extra_id_0>', '<extra_id_1>']
        })
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.lr = args.lr
        # self.acc = MyAccuracy(device=torch.device("cuda"))
        self.acc = MyAccuracy()
        self.mode = args.mode

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]
        logits = outputs[1]
        preds = torch.argmax(logits, 1)
        labels = batch['labels']
        correct = (preds == labels).sum()
        total = labels.numel()
        acc = self.acc(correct, total)
        self.log('val_loss', loss)
        self.log('val_acc',
                 self.acc,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        if self.mode != 'infer':
            loss = outputs[0]
            logits = outputs[1]
            preds = torch.argmax(logits, 1)
            labels = batch['labels']
            limit = 0.05
            # probs = torch.softmax(logits, dim=-1)
            # idx = probs[:, 1] < limit
            # correct = (preds[idx] == labels[idx]).sum()
            # total = idx.sum()
            correct = (preds == labels).sum()
            total = labels.numel()
            self.acc(correct, total)
            self.log('test_loss', loss)
            self.log('test_acc', self.acc, on_step=False, on_epoch=True)
            return loss
        else:
            logits = outputs[0]
            preds = torch.argmax(logits, 1)
            correct = preds.sum()
            total = preds.numel()
            acc = self.acc(correct, total)
            # print(acc)
            self.log('test_acc', self.acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.lr)


@dataclass
class Arguments:
    train_data_path: Optional[str] = field(default='./data/train.json')
    val_data_path: Optional[str] = field(default='./data/val.json')
    test_data_path: Optional[str] = field(default='./data/test.json')
    lr: Optional[float] = field(default=1e-5)
    gpus: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=2021)
    epochs: Optional[int] = field(default=8)
    train_batch_size: Optional[int] = field(default=4)
    valtest_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    mode: Optional[str] = field(default='train')
    ckpt_path: Optional[str] = field(
        default='./logs/bert/lightning_logs/version_1/checkpoints')
    pretrain_path: Optional[str] = field(default='hfl/chinese-bert-wwm-ext')
    pl_root_dir: Optional[str] = field(default='./logs/bert')

    def __post_init__(self):
        if self.mode != 'train':
            if os.path.exists(self.ckpt_path):
                names = os.listdir(self.ckpt_path)
                self.ckpt_path = os.path.join(self.ckpt_path, names[0])


class ModelAPI:
    def __init__(self, device):
        parser = HfArgumentParser(Arguments)
        args, = parser.parse_args_into_dataclasses()
        args.mode = 'test'
        args.__post_init__()
        self.args = args
        model = PLModel(args)
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        # print(ckpt.keys())
        model.load_state_dict(ckpt['state_dict'])
        self.tokenizer = model.tokenizer
        self.model = model.model
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, sents):
        """
            sents: [[essay, word], ...]
        """
        res = self.tokenizer(sents,
                             return_tensors='pt',
                             padding=True,
                             max_length=512,
                             truncation=True)
        res = self.batch_to_device(res)
        outputs = self.model(**res, return_dict=True)
        logits = outputs.logits
        # print(logits.size())
        scores = torch.softmax(logits, dim=-1)
        return scores[:, 1].cpu().numpy()
    
    def batch_to_device(self, batch, device=None):
        if device is None:
            device = self.device
        for k in batch:
            batch[k] = batch[k].to(device)
        return batch

def main(args):
    pl.seed_everything(args.seed)
    if args.mode == 'train':
        model = PLModel(args)
        pld = PLDataLoader(args, model.tokenizer)
        checkpoint_callback = ModelCheckpoint(monitor='val_acc',
                                              save_top_k=1,
                                              mode='max',
                                              verbose=True)
        earlystop_callback = EarlyStopping(monitor='val_acc',
                                           verbose=True,
                                           mode='max')
        trainer = pl.Trainer(
            gpus=args.gpus,
            callbacks=[checkpoint_callback, earlystop_callback],
            accelerator='ddp',
            # limit_train_batches=0.001,
            # limit_val_batches=0.01,
            gradient_clip_val=1.0,
            max_epochs=args.epochs,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            default_root_dir=args.pl_root_dir)
        trainer.fit(model, pld)
    elif args.mode == 'test':
        model = PLModel(args)
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        # print(ckpt.keys())
        model.load_state_dict(ckpt['state_dict'])
        pld = PLDataLoader(args, model.tokenizer)
        trainer = pl.Trainer(gpus=args.gpus,
                             max_epochs=args.epochs,
                             default_root_dir=args.pl_root_dir)
        trainer.test(model, datamodule=pld)


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    args, = parser.parse_args_into_dataclasses()
    main(args)
