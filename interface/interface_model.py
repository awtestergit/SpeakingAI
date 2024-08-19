# -*- coding: utf-8 -*-
#
# @author awtestergit
# @description interface classes
#
import numpy
import torch
from typing import Iterator

"""
IBaseModel, interface to wrap model
"""
class IBaseModel():
    ### embedding model
    EMBED_SIZE = 1024 # embedding size of the model
    MAX_LENGTH = 8000 # max seq length, e.g, openai 8192
    def __init__(self) -> None:
        self.device = self.__get_device__()

    def __get_device__(self):
        _device = None
        # check if cuda
        if torch.cuda.is_available():
            _device = torch.device('cuda')
        else:# Check that MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                        "built with MPS enabled.")
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine.")
            else:
                _device = torch.device("mps")
        if _device is None:
            _device = torch.device('cpu')
        
        return _device

"""
IRerankerModel, interface to wrap model
"""
class IRerankerModel(IBaseModel):
    ### embedding model
    MAX_LENGTH = 4000 # max seq length, e.g., Cohere 4096
    def __init__(self, model_path, threshod:float = 0.5) -> None:
        super().__init__()
        self.model_path = model_path
        self.threshold = threshod

    def rerank_score(self, inputA:str, inputB:list)->tuple[bool, float]:
        raise NotImplementedError("IRerankerModel rerank score not implemented")

"""IEmbeddingModel
"""
class IEmbeddingModel(IBaseModel):
    ### embedding model
    EMBED_SIZE = 1024 # embedding size of the model
    MAX_LENGTH = 8000 # max seq length, openai 8192
    def __init__(self) -> None:
        super().__init__()

    def encode(self, inputs, to_list:bool=False, *args):
        raise NotImplementedError("ILanguageModel base class encode")

"""
ILanguageModel, interface to wrap model
"""
class ILanguageModel(IBaseModel):
    def __init__(self, max_context_length, token_ex, seed) -> None:
        super().__init__()
        self.MAX_LENGTH = max_context_length # max token lengths
        self.TOKEN_EX = token_ex # 
        self.seed = seed

    def generate(self, inputs, splitter='', stop=[], replace_stop=True, **kwargs):
        raise NotImplementedError("ILanguageModel base class generate")

    def chat(self, inputs, history=[], splitter='', stop=[], replace_stop=True, **kwargs):
        raise NotImplementedError("ILanguageModel base class generate")

    def stream_generate(self, inputs, splitter='', stop=[], replace_stop=True, output_delta=False, **kwargs):
        raise NotImplementedError("ILanguageModel base class stream generate")

    def stream_chat(self, inputs, history=[], splitter='', stop=[], replace_stop=True, output_delta=False, **kwargs):
        raise NotImplementedError("ILanguageModel base class stream chat")


"""ITTSModel
    tts model wrapper
"""
class ITTSModel(IBaseModel):
    def __init__(self) -> None:
        pass

    def text_to_data(self, input_text:str | Iterator, sample_rate=16000, **kwargs) -> tuple[int, str, numpy.array]:
        raise NotImplementedError("ITTSModel base class, text_to_data")



"""IASRModel
    ASR model wrapper
"""
class IASRModel(IBaseModel):
    def __init__(self) -> None:
        pass

    def voice_to_text(self, data, rate, bit_depth, target_sample_rate, **kwargs) -> str:
        raise NotImplementedError("IASRModel base, voice to text")
