# -*- coding: utf-8 -*-
#
# @author awtestergit
# @description Reranker
#

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from interface.interface_model import IRerankerModel

class BAAIReRankerM3Model(IRerankerModel):
    ### embedding model
    EMBED_SIZE = 1024 # embedding size of the model
    MAX_LENGTH = 2048 # max seq length 8192 of M3

    def __init__(self, model_path="./baai_reranker", threshod:float = 0.5, embed_size=1024, seq_size=2048):
        super().__init__(model_path=model_path, threshod=threshod)
        self.EMBED_SIZE = embed_size
        self.MAX_LENGTH = seq_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16, local_files_only=True)#.to(self.device)
        self.model.eval()

    def rerank_score(self, inputA:str, inputB:str)->tuple[bool, float]:
        """
        rank the similarity score between inputA and inputB
        output: True, 0~1 score
        """
        # Compute token embeddings
        with torch.no_grad():
            # Tokenize sentences
            input_pairs = [[inputA, inputB]]
            encoded_input = self.tokenizer(input_pairs, padding=True, max_length=self.MAX_LENGTH, truncation=True, return_tensors='pt').to(self.device)
            scores = self.model(**encoded_input, return_dict=True).logits.view(-1,).float()
            # convert to 0-1
            scores = torch.nn.functional.sigmoid(scores)
        return True, scores.item()
