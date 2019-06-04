import numpy as np


class SWEM():
    """
    Simple Word-Embeddingbased Models (SWEM)
    https://arxiv.org/abs/1805.09843v1
    """

    def __init__(self, ft_model, oov_initialize_range=(-0.01, 0.01)):
        self.ft_model = ft_model
        self.vocab = self.ft_model.get_words()
        self.embedding_dim = self.ft_model.get_dimension()
        self.oov_initialize_range = oov_initialize_range

        if self.oov_initialize_range[0] > self.oov_initialize_range[1]:
            raise ValueError("Specify valid initialize range: "
                             f"[{self.oov_initialize_range[0]}, {self.oov_initialize_range[1]}]")

    def get_word_embeddings(self, tokens):
        vectors = []
        for token in tokens:
            vectors.append(self.ft_model.get_word_vector(token))
            # if word in self.vocab:
            #     vectors.append(self.w2v[word])
            # else:
            #     vectors.append(np.random.uniform(self.oov_initialize_range[0],
            #                                      self.oov_initialize_range[1],
            #                                      self.embedding_dim))
        return np.array(vectors)

    def average_pooling(self, tokens):
        word_embeddings = self.get_word_embeddings(tokens)
        return np.mean(word_embeddings, axis=0)

    def max_pooling(self, tokens):
        word_embeddings = self.get_word_embeddings(tokens)
        return np.max(word_embeddings, axis=0)

    def concat_average_max_pooling(self, tokens):
        word_embeddings = self.get_word_embeddings(tokens)
        return np.r_[np.mean(word_embeddings, axis=0), np.max(word_embeddings, axis=0)]

    def hierarchical_pooling(self, tokens, n):
        word_embeddings = self.get_word_embeddings(tokens)

        tokens_len = word_embeddings.shape[0]
        if n > tokens_len:
            raise ValueError(f"window size must be less than tokens length\
                               / window_size:{n} tokens_length:{tokens_len}")
        window_average_pooling_vec = [np.mean(word_embeddings[i:i + n], axis=0) for i in range(tokens_len - n + 1)]

        return np.max(window_average_pooling_vec, axis=0)
