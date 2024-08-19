# -*- coding: utf-8 -*-
#
# @author awtestergit
# @description interface to TTS, ASR and LLM, for digital human applications
#
import numpy as np
from typing import Iterator
from pydantic import BaseModel, Field, ConfigDict
import json
from interface.interface_model import ILanguageModel, IASRModel, ITTSModel, IRerankerModel

class ITALModel():
    def __init__(self, llm:ILanguageModel, reranker:IRerankerModel, asr:IASRModel, tts:ITTSModel) -> None:
        self.llm = llm
        self.reranker = reranker
        self.asr = asr
        self.tts = tts
        self.history = [] # chat history
    
    def audio_in(self, audio:np.ndarray, rate:int=0, language:str=None, **kwargs)->str:
        """
        input: audio
            rate: sample_rate
            language: zh/en for example
        output: text
        """
        raise NotImplementedError

    def audio_out(self, inputs:str|Iterator, **kwargs)->tuple[int, int, np.ndarray]:
        """
        input: text | Iterator to be used with streaming
        output: sample_rate, channels, np.ndarray audio
        """
        raise NotImplementedError

    def query(self, inputs:str, **kwargs)->tuple[Iterator, str]:
        """
        query to ask LLM - either from FAQ, or LLM answers
        """
        raise NotImplementedError

#@dataclass
class TALAudioRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) # allow numpy array
    audioType:str = 'int16' # only allows 'int16' or 'int32', or 'float32'
    bitDepth:int = 16 # audio bit depth, only allows 16 (if audio type is int16) or 32 if int32
    #audioFile:np.ndarray = Field(default_factory=lambda: np.zeros(10)) # audio bytes
    sampleRate:int = 16000 # 16000
    channels:int = 1 #
    audioFormat:str = 'wav' #wav
    question:str = '' # question
    answer:str = '' #answer
    url:str = '' # action url
    sign:int = 1 # 1 start; 2 middle; 3 end
    order:int = 1 # increment by 1

#@dataclass
class TALAudioResponse(TALAudioRequest):
    answer:str = '' #answer
    sign:int = 1 # 1 start; 2 middle; 3 end
    order:int = 1 # increment by 1

def tal_encoder(tal:TALAudioRequest, wav:bytes=b'', byte_length = 4)->bytes:
    """
    encode tal and wav into a byte sequence, format as:
        total_length|header_length|header|wav, where header is the tal object in bytes
    byte_length: both total_length and header_length are 4 bytes, 
        total_length includes total length of 'total_length|header_length|header|wav'
        header_length indicating the length of 'header_length|header' of the byte sequence
    the rest is the wav bytes
    tal: TALAudioRequest or TALAudioResponse
    wav: wav bytes
    output: the byte sequence in 'total_length|header_length|header|wav'
    """
    tal_data = json.dumps(tal.__dict__) # dump to json string
    tal_data = tal_data.encode(errors='ignore') # encode into bytes
    data_size = len(tal_data) #tal data is a byte string, so length is len(), not sizeof()
    #byte_order = sys.byteorder
    byte_order = 'little' #fix to little between client and server
    data_size_bytes = data_size.to_bytes(byte_length, byteorder=byte_order) #byteorder to consider
    header = data_size_bytes + tal_data # concat bytes
    header_length = byte_length + data_size # total header size, including the 4 bytes
    wav_length = len(wav) #sys.getsizeof(wav)

    print(f"wav sizeof: {wav_length} vs. len(): {len(wav)}")

    total_length = byte_length + header_length + wav_length # including the 4 byets, now the length is total of 'total_length|header_length|header|wav'
    total_length_bytes = total_length.to_bytes(byte_length, byteorder=byte_order)
    output = total_length_bytes + header + wav #concat
    return output

def tal_decoder(obj:bytes, byte_length=4)->None | tuple[(dict, bytes), any]:
    """
    input: bytes in the format of 'total_size|header_size|header|wav'
        decoder extract the header and wav
    output: ([dict, wav (in bytes)], rest_bytes), (dict, wav) is the first chunk of wav, rest_bytes is the rest object if any
        None if wav is only partial data
    """
    whole_length = len(obj) # whole length

    #byte_order = sys.byteorder
    byte_order = 'little' #fix to little between client and server
    total_size = obj[:byte_length] # total size of this wav data incidated by header
    total_size = int.from_bytes(total_size, byteorder=byte_order)

    # check total size vs whole length
    if whole_length < total_size:
        return None

    header_size = obj[byte_length:byte_length*2] #two byte_length
    header_size = int.from_bytes(header_size, byteorder=byte_order)# byteorder to consider
    header_size += byte_length*2 # the end of header in the byte sequence
    header = obj[byte_length*2:header_size] # bytes of tal object, starting at byte_length, end at header_size
    #print(header_size)
    #print(header)
    header = header.decode() # decode into str
    #print()
    #print(header)
    header = json.loads(header) # to dict
    wav = obj[header_size:total_size] #the rest is wav bytes

    # if whole length > total size, if nothing left, return b''
    rest_bytes = obj[total_size:] if (whole_length > total_size) else b''

    print(f"total_size: {total_size}, whole length: {whole_length}, rest bytes length: {len(rest_bytes)}")

    return ((header, wav), rest_bytes)
