from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

class BPETokenizer:
    def __init__(self, sentence_list):
        """
        sentence_list - список предложений для обучения
        """
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.enable_truncation(max_length=15)
        self.tokenizer.enable_padding(length=15)
        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        self.tokenizer.train_from_iterator(sentence_list, trainer)

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        return self.tokenizer.encode(sentence).ids


    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        return self.tokenizer.decode(token_list).split()