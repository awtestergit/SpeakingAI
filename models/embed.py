# -*- coding: utf-8 -*-
#
# @author awtestergit
# @description embedding model
#

from transformers import AutoTokenizer, AutoModel

import torch
from interface.interface_model import IEmbeddingModel

class IntfloatEmbeddingModel(IEmbeddingModel):
    ### embedding model
    EMBED_SIZE = 1024 # embedding size of the model
    MAX_LENGTH = 512 # max seq length    

    def __init__(self, model_path, embed_size=1024, seq_size=512):
        super().__init__()
        self.EMBED_SIZE = embed_size
        self.MAX_LENGTH = seq_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)

        e = self.encode("hello")
        assert e.shape[-1] == self.EMBED_SIZE

    def encode(self, inputs, to_list=False):
        """
        per model card, Here are some rules of thumb:
        Use "query: " and "passage: " correspondingly for asymmetric tasks such as passage retrieval in open QA, ad-hoc information retrieval.
        Use "query: " prefix for symmetric tasks such as semantic similarity, bitext mining, paraphrase retrieval.
        Use "query: " prefix if you want to use embeddings as features, such as linear probing classification, clustering.
        """
        # add query: prefix
        inputs = f"query: {inputs}"
        # Compute token embeddings
        with torch.no_grad():
            # Tokenize sentences
            encoded_input = self.tokenizer(inputs, padding=True, max_length=self.MAX_LENGTH, truncation=True, return_tensors='pt').to(self.device)
            model_output = self.model(**encoded_input)
        # Perform pooling. In this case, mean pooling.
        last_hidden_state = model_output.last_hidden_state
        sentence_embeddings = self.__mean_pooling__(last_hidden_state, encoded_input['attention_mask'])
        result = sentence_embeddings.squeeze()
        result = result.tolist() if to_list else result
        return result

    # Mean Pooling - Take attention mask into account for correct averaging
    def __mean_pooling__(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        # normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

