# -*- coding: utf-8 -*-

#
# @author awtestergit
# @description TTS models
#

import numpy
import torch
from transformers import BarkModel, AutoProcessor, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from interface.interface_model import ITTSModel
from anbutils.utilities import resample_data

class BarkTTS(ITTSModel):
    def __init__(self, model_path:str, speaker:str) -> None:
        self.model_path = model_path
        self.speaker = speaker # voice preset
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.model = BarkModel.from_pretrained(self.model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def text_to_data(self, input_text: str, sample_rate=16000, voice_preset="v2/en_speaker_0", **kwargs) -> tuple[int, str, numpy.array]:
        """
        convert input text to audio data
        text: the input text
        sample_rate: the sample_rate of output data
        voice_preset: the preset voice, under v2 subfolder, 
            for all speakers, check https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
        output: [sample_rate, dtype, audio_data] in int, numpy.float32, numpy.array
        """
        
        voice_preset = voice_preset if voice_preset is not None else self.speaker
        input_processed = self.processor(input_text, voice_preset=voice_preset).to(self.device)

        audio_array = self.model.generate(**input_processed)
        audio_array = audio_array.cpu().numpy().squeeze()
        rate = self.model.generation_config.sample_rate
        if rate != sample_rate:
            sample_rate, audio_array = resample_data(rate, audio_array, sample_rate)
        
        dtype = 'float32'
        return (sample_rate, dtype, audio_array)
    

class Speecht5TTS(ITTSModel):
    def __init__(self, speaker_index:int=1350) -> None:
        super().__init__()
        self.model_name = "microsoft/speecht5_tts"
        self.model_hifigan = "microsoft/speecht5_hifigan"
        self.speaker_dataset = "Matthijs/cmu-arctic-xvectors"
        self.speaker_index = speaker_index
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    def text_to_data(self, input_text: str, sample_rate=16000, speaker_index=-1, **kwargs) -> tuple[int, str, numpy.array]:
        """
        convert input text to audio data
        text: the input text
        sample_rate: the sample_rate of output data
        speaker_index: the index of the dataset:
            https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors
        output: [sample_rate, dtype, audio_data] in int, numpy.float32, numpy.array
        """
        speaker_index = speaker_index if speaker_index > 0 else self.speaker_index

        inputs = self.processor(text=input_text, return_tensors="pt").to(self.device)
        # load xvector containing speaker's voice characteristics from a dataset
        embeddings_dataset = load_dataset(self.speaker_dataset, split="validation")
        #speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        speaker_embeddings = torch.tensor(embeddings_dataset[1350]["xvector"]).unsqueeze(0)

        audio_array =self.model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder)

        audio_array = audio_array.cpu().numpy().squeeze()
        rate = self.processor.feature_extractor.sampling_rate
        if rate != sample_rate:
            sample_rate, audio_array = resample_data(rate, audio_array, sample_rate)
        
        dtype = 'float32'
        return (sample_rate, dtype, audio_array)
