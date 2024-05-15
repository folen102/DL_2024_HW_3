import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, Adafactor
import src.metrics as metrics

class Seq2SeqT5(nn.Module):
    def __init__(self, device, pretrained_model_name, tokenizer, learning_rate):
        super(Seq2SeqT5, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.t5_model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name).to(device)
        self.t5_model.resize_token_embeddings(len(tokenizer))
        self.optimizer = Adafactor(self.t5_model.parameters(), lr=learning_rate, relative_step=False)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_tensor, attention_mask, labels=None):
        input_tensor = input_tensor.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
            return self.t5_model(input_ids=input_tensor, attention_mask=attention_mask, labels=labels)
        return self.t5_model(input_ids=input_tensor, attention_mask=attention_mask)

    def training_step(self, batch):
        self.optimizer.zero_grad()
        input_tensor, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        outputs = self.forward(input_tensor, attention_mask, labels)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self, batch):
        with torch.no_grad():
            input_tensor, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            outputs = self.forward(input_tensor, attention_mask, labels)
            loss = outputs.loss
        return loss.item()

    def predict(self, input_tensor, attention_mask):
        self.eval()
        with torch.no_grad():
            generated_ids = self.t5_model.generate(input_ids=input_tensor.to(self.device), attention_mask=attention_mask.to(self.device))
        return generated_ids

    def eval_bleu(self, predicted_ids, target_tensor):
        predicted = predicted_ids.squeeze().detach().cpu().numpy()[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(predicted=predicted, actual=actuals, target_tokenizer=self.tokenizer)
        return bleu_score, actual_sentences, predicted_sentences