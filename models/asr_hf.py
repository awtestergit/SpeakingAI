# -*- coding: utf-8 -*-
"""
@author: awtestergit
@description: ASR model, modified from parrots, author:XuMing(xuming624@qq.com)
"""
from typing import Optional, Union

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from interface.interface_model import IASRModel
from anbutils.utilities import convert_audio_int_bytes_to_numpy, convert_audio_numpy_int_to_float, resample_data

class ASRWhisperModel(IASRModel):
    def __init__(
            self,
            model_name_or_path: str,
            max_new_tokens: Optional[int] = 128,
            chunk_length_s: Optional[int] = 15,
            batch_size: Optional[int] = 16,
            torch_dtype: Optional[str] = "float16",
            low_cpu_mem_usage = False, # add as a parameter - awtestergit
            use_flash_attention_2: Optional[bool] = False,
            language: Optional[str] = "en",
            initial_prompt:str = "",
            sampling_rate:int = 16000, # default sampling rate
            **kwargs
    ):
        """
        Initialize the speech recognition object.
        :param model_name_or_path: Model name or path, like:
            'BELLE-2/Belle-distilwhisper-large-v2-zh', 'distil-whisper/distil-large-v2', ...
            model in HuggingFace Model Hub and release from
        :param use_cuda: Whether or not to use CUDA for inference.
        :param cuda_device: Which cuda device to use for inference.
        :param max_new_tokens: The maximum number of new tokens to generate, ignoring the number of tokens in the
            prompt.
        :param chunk_length_s: The length in seconds of the audio chunks to feed to the model.
        :param batch_size: The batch size to use for inference.
        :param torch_dtype: The torch dtype to use for inference.
        :param use_flash_attention_2: Whether or not to use the FlashAttention2 module.
        :param language: The language of the model to use.
        :param initial_prompt: the initial prompt for the language. e.g, "transcribe the audio into texts"
        :param sampling_rate: 16000, which is Whisper was trained on
        :param kwargs: Additional keyword arguments passed along to the pipeline.
        """
        self.device = self.__get_device__()

        torch_dtype = (
            torch_dtype
            if torch_dtype in ["auto", None]
            else getattr(torch, torch_dtype)
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_flash_attention_2=use_flash_attention_2,
        )
        self.model.to(self.device)

        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)
        self.language = language
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.chunk_length_s = chunk_length_s
        self.sampling_rate = sampling_rate
        self.torch_dtype = torch_dtype
        self.initial_prompt = initial_prompt
        self.initial_prompt_ids = self.processor.get_prompt_ids(initial_prompt, return_tensors='pt').to(self.device)
    def predict(self, audio: np.ndarray, sample_rate:int=16000):
        if sample_rate != self.sampling_rate:
            sample_rate, audio = resample_data(sample_rate, audio)
        
        inputs = self.processor.feature_extractor(raw_speech=audio, return_attention_mask=True, return_tensors='pt', sampling_rate=sample_rate).to(self.device)
        inputs = inputs.to(self.torch_dtype) # proper dtype
        generated_ids = self.model.generate(inputs=inputs.input_features, attention_mask=inputs.attention_mask, max_new_tokens=self.max_new_tokens, language=self.language, prompt_ids=self.initial_prompt_ids)
        start = len(self.initial_prompt_ids)
        ### not batch processing
        generated_ids = generated_ids[0][start:]
        transcription = self.processor.decode(generated_ids, skip_special_tokens=True)
        return transcription
   
    def voice_to_text(self, data:bytes|np.ndarray, rate=16000, bit_depth:int=16, target_sample_rate=16000) -> str:
        """
        data: bytes or np array
        """
        # if raw bytes, convert to numpy float, in dict
        if isinstance(data, bytes):
            data = convert_audio_int_bytes_to_numpy(data=data, sample_rate=rate,bit_depth=bit_depth)
        elif isinstance(data, np.ndarray):
            data = convert_audio_numpy_int_to_float(data=data, sample_rate=rate)
        else:
            raise ValueError(f"voice to text: input data must be either raw bytes or numpy array, but is instead {type(data)}")
        sample_rate = data['sampling_rate']
        audio = data['raw']
        if sample_rate != target_sample_rate:
            sample_rate, audio = resample_data(sample_rate, data=audio, target_rate=target_sample_rate)
        result = self.predict(audio=audio, sample_rate=sample_rate)
        return result
