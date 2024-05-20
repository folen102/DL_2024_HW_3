import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer
from src.data.utils import TextUtils, short_text_filter_function

class T5Dataset(Dataset):
    def __init__(self, input_texts, target_texts, source_tokenizer, target_tokenizer, max_length):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        source = self.source_tokenizer(input_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        target = self.target_tokenizer(target_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': source.input_ids.squeeze(),
            'attention_mask': source.attention_mask.squeeze(),
            'labels': target.input_ids.squeeze()
        }

class T5DataManager:
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def prepare_data(self):
        pairs = TextUtils.read_langs_pairs_from_file(filename=self.config["filename"])
        prefix_filter = self.config['prefix_filter']
        if prefix_filter:
            prefix_filter = tuple(prefix_filter)

        input_texts, target_texts = [], []
        for pair in pairs:
            input_text, target_text = pair[0], pair[1]
            if short_text_filter_function(pair, self.config['max_length'], prefix_filter):
                input_texts.append(input_text)
                target_texts.append(target_text)

        train_size = int(len(input_texts) * self.config["train_size"])
        train_input_texts, val_input_texts = input_texts[:train_size], input_texts[train_size:]
        train_target_texts, val_target_texts = target_texts[:train_size], target_texts[train_size:]

        self.source_tokenizer = T5Tokenizer.from_pretrained(self.config['pretrained_model_name'])
        self.target_tokenizer = T5Tokenizer.from_pretrained(self.config['pretrained_model_name'])

        train_dataset = T5Dataset(train_input_texts, train_target_texts, self.source_tokenizer, self.target_tokenizer, self.config['max_length'])
        val_dataset = T5Dataset(val_input_texts, val_target_texts, self.source_tokenizer, self.target_tokenizer, self.config['max_length'])

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.config["batch_size"])
        val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=self.config["batch_size"], drop_last=True)

        return train_dataloader, val_dataloader

    def adjust_tokenizer(self):
        pairs = TextUtils.read_langs_pairs_from_file(filename=self.config["filename"])
        input_texts, target_texts = zip(*pairs)
        return list(input_texts) + list(target_texts)
