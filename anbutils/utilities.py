import numpy as np
from scipy.signal import resample

###audios
def resample_wav(wav, target_rate:int=16000)->tuple[int,any]:
    """
    input: a wav format (sample_rate, data)
    output: wav format (16000, data)
    """
    rate, data = wav
    return resample_data(rate=rate, data=data, target_rate=target_rate)

def resample_data(rate, data, target_rate:int=16000):
    
    # if stereo
    if len(data.shape)== 2: #(samples, 2)
        data = data.T[0] #get mono

    if rate != target_rate: # resample
        new_rate = target_rate
        data = resample(data, int(len(data)*new_rate/rate))
        rate = new_rate

    if data.dtype != np.float32:
        data = data.astype(np.float32) # convert to float32
        data /= np.max(np.abs(data))

    return rate, data

def convert_audio_int_bytes_to_numpy(data, sample_rate, bit_depth:int)->dict:
    """
    given a raw audio data of int bytes, convert to ndarray of float
    output: dict of {'sampling_rate': sample_rate, 'raw': data_np}
    """
    dtype = 0
    if bit_depth == 16:
        dtype = np.int16
    elif bit_depth == 32:
        dtype = np.int32
    else:
        raise ValueError(f"convert audio int bytes: bit depth must be either 16 or 32, input as: {bit_depth}")
    sample_width = int(bit_depth/8)
    data_np = np.frombuffer(data, dtype=dtype, count=len(data)//sample_width, offset=0) #
    data_np = data_np * (0.5**15) # to float
    data_np = data_np.astype(np.float32)
    return {
        'sampling_rate': sample_rate, 
        'raw': data_np
    }

def convert_audio_numpy_int_to_float(data:np.ndarray, sample_rate)->dict:
    """
    given a numpy data array, convert to ndarray of float
    output: dict of {'sampling_rate': sample_rate, 'raw': data_np}
    """
    if len(data)==0:
        return {}

    dtype = 0
    if not isinstance(data, np.ndarray):
        raise ValueError(f"convert audio numpy to float: data is not ndarray, but is {type(data)}")
    if 'int' not in str(type(data[0])).lower():
        raise ValueError(f"convert audio numpy int to float: data is not numpy int")
    
    data_np = data * (0.5**15) # to float, or / 32767
    data_np = data_np.astype(np.float32)
    return {
        'sampling_rate': sample_rate, 
        'raw': data_np
    }

def convert_audio_float32_to_int16(data:np.ndarray, to_bytes:bool=False)->np.ndarray|bytes:
    """
    convert np array of floats to np array of int, or bytes if to_bytes
    """
    if len(data)==0:
        return None
    #dtype = np.int32 if bit_depth==32 else np.int16
    dtype = np.int16
    data_type = str(type(data[0])).lower()
    if 'float' in data_type:
        data = data.clip(min=-1, max=1) # clip to -1 and 1
        data[data<0.0] *= 32768.0 # if negative
        data[data>0.0] *= 32767.0 # if positive
        data = np.floor(data).astype(dtype=dtype) # floor, to int
    elif 'int' in data_type:
        data = data.astype(dtype=dtype)
    else:
        raise ValueError(f"convert audio to int: data type should be either float or int, instead is: {data_type}")
    
    data = data if not to_bytes else data.tobytes()

    return data