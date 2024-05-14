import math
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.optim.lr_scheduler import StepLR
from torch.nn import Transformer
from src.models.positional_encoding import PositionalEncoding
import src.metrics as metrics

class PositionalEncoding(nn.Module):
    """ Добавление позиционных энкодеров к входным эмбеддингам. """
    def __init__(self, device, embedding_size, dropout, maxlen=15):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros(maxlen, embedding_size)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.register_buffer('pos_embedding', pos_embedding.unsqueeze(0).to(device))

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])


class Seq2SeqTransformer(nn.Module):
    def __init__(self, device, embedding_size, num_encoder_layers, dim_feedforward, src_voc_size, trg_voc_size,
                 target_tokenizer, source_tokenizer, lr_decay_step, lr=1e-4, lr_decay=0.1, dropout_rate=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.device = device
        self.special_tokens = {"pad": 0, "bos": 1, "eos": 2}

        self.src_emb = nn.Embedding(src_voc_size, embedding_size).to(device)
        self.trg_emb = nn.Embedding(trg_voc_size, embedding_size).to(device)
        self.positional_encoding = PositionalEncoding(device, embedding_size, dropout_rate)

        self.transformer = Transformer(d_model=embedding_size, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=0, dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                       batch_first=True).encoder.to(device)

        self.vocab_projection_layer = nn.Linear(embedding_size, trg_voc_size).to(device)
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=lr_decay_step, gamma=lr_decay)

        self.target_tokenizer = target_tokenizer
        self.source_tokenizer = source_tokenizer

    def create_masks(self, src):
        src_mask = torch.zeros((src.shape[1], src.shape[1]), device=self.device).type(torch.bool)
        src_pad_mask = (src == self.special_tokens["pad"])
        return src_mask, src_pad_mask

    def forward(self, src):
        src_mask, src_pad_mask = self.create_masks(src)
        src_emb = self.positional_encoding(self.src_emb(src))
        out = self.transformer(src_emb, src_mask, src_pad_mask)
        logits = self.vocab_projection_layer(out)
        preds = torch.argmax(logits, dim=2)
        return preds, logits

    def training_step(self, batch):
        self.optimizer.zero_grad()
        _, logits = self.forward(batch[0])
        loss = self.loss(logits.view(-1, logits.shape[-1]), batch[1].view(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self, batch):
        with torch.no_grad():
            _, logits = self.forward(batch[0])
            loss = self.loss(logits.view(-1, logits.shape[-1]), batch[1].view(-1))
        return loss.item()

    def predict(self, sentences):
        src_tokenized = torch.tensor([self.source_tokenizer(s) for s in sentences]).to(self.device)
        preds = self.forward(src_tokenized)[0].cpu().detach().numpy()
        return [self.target_tokenizer.decode(i) for i in preds]

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.detach().cpu().numpy()
        actuals = target_tensor.detach().cpu().numpy()
        return metrics.bleu_scorer(predicted, actuals, self.target_tokenizer)
