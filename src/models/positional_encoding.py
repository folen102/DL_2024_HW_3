import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size, maxlen):
        """
        emb_size - размер эмбеддингов
        maxlen - длинна контекста
        """
        super(PositionalEncoding, self).__init__()
        # TODO: Реализуйте конструтор
        pass


    def forward(self, token_embedding):
        """
        token_embedding - тензор матрицы эмбеддингов
        """
        # TODO: Реализуйте сложение эмбединнгов токенов с позиционными эмбеддингами
        pass