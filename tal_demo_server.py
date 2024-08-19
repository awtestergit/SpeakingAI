# -*- coding: utf-8 -*-
#
# @author awtestergit
# @description TTS, ASR and LLM, for digital human applications, using parrots TTS & ASR
#
from loguru import logger
from argparse import ArgumentParser
import json
from pydantic import TypeAdapter
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient

from models.tal_model import TALModel
from models.llm_hf import Llama3
from models.reranker import BAAIReRankerM3Model
from models.embed import IntfloatEmbeddingModel
from models.asr_hf import ASRWhisperModel # asr
from interface.interface_tal import TALAudioRequest, tal_decoder
from models.ttsmodel import BarkTTS, Speecht5TTS #tts
from qdrantclient_vdb.qdrant_knowledge import KnowledgeWarehouse

DEFAULT_SAMPLE_RATE = 16000 # default audio sample reate
MAX_AUDIO_LENGTH = 16000*60*32/8 #60 seconds, 16000 sample_rate, 32-bit

# app factory
def create_app():
    config_file = "config.json"
    g_config = {}
    with open(config_file) as f:
        g_config = json.load(f)

    # initialize models
    llm = None
    llm_path = g_config['LLM']
    logger.info(f"Load LLM model: {llm_path}")
    llm = Llama3(llm_path)
    #reranker
    reranker = None
    reranker_path = g_config['RERANKER']
    logger.info(f"Load reranker model: {reranker_path}")
    reranker = BAAIReRankerM3Model(reranker_path)
    # embedding
    embedding_model = None
    embedding_path = g_config['EMBEDDING']
    logger.info(f"Load embedding model: {embedding_path}")
    embedding_model = IntfloatEmbeddingModel(embedding_path)
    # ASR
    asr = None
    asr_path = g_config['ASR']
    logger.info(f"Load ASR model: {asr_path}")
    asr_prompt = "Transcribe the audio into texts."
    asr = ASRWhisperModel(asr_path, initial_prompt=asr_prompt)
    # TTS
    tts = None
    tts_path = g_config['TTS']
    # tts_speaker = g_config['SPEAKER']
    # logger.info(f"Load TTS model: {tts_path}, speaker: {tts_speaker}")
    # tts = BarkTTS(tts_path, speaker=tts_speaker)
    tts = Speecht5TTS()

    # ai bot name
    ai_name=g_config['AINAME']

    #vdb
    vdb_ip = g_config['VDBIP']
    vdb_port = g_config['VDBPORT']
    collection_name = g_config['VDBNAME']
    vdb_conf = g_config['VDBCONF']
    top_k = g_config['TOP']
    client = QdrantClient(vdb_ip, port=vdb_port)
    kw_vdb = KnowledgeWarehouse(client, collection_name, embedding_model)

    # tal
    g_tal = TALModel(llm=llm, reranker=reranker, asr=asr, tts=tts, kw_vdb=kw_vdb, ai_name=ai_name, vdb_conf=vdb_conf)

    # chat history, for multi-users, this should be a session variable instead
    history = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # print out client connection
        print("************************************************")
        print("************************************************")
        print("***TAL Demo Server***")
        print("************************************************")
        print("************************************************")
        yield
        # Clean up if any

    app = FastAPI(lifespan=lifespan, debug=True)

    @app.get('/')
    def index():
        return "None"
    
    @app.post('/audio_in')
    async def audio_in(request: Request):
        # new query from user, 
        # get the data object from request
        data = await request.body()
        (r, wav), rest_bytes = tal_decoder(data)
        # convert to talrequest
        ta = TypeAdapter(TALAudioRequest)
        req = ta.validate_python(r)
        query = ''

        # check if text
        sample_rate = req.sampleRate
        bit_depth = req.bitDepth
        if req.question is None or len(req.question)==0: #need to do TTS
            query = g_tal.audio_in(audio=wav, rate=sample_rate, bit_depth=bit_depth, target_sample_rate=DEFAULT_SAMPLE_RATE)
        else:
            query = req.question
        
        print(query)

        _length = g_config['LENGTH']
        kwargs = {
            'temperature': 0.3,
            'max_new_tokens': _length,
            'top_k': 3,
            'top_p': 0.95,
            'repetition_penalty': 1.1,
        }
        # query
        answer_iter, sources = g_tal.query(query=query, history=history, top=top_k, conf=vdb_conf, **kwargs)

        # audio_out
        r_iter = g_tal.tal_audio_response(query=query, answers=answer_iter, history=history, sample_rate=DEFAULT_SAMPLE_RATE)
        # stream response
        return StreamingResponse(r_iter,media_type='application/octet-stream')

    return app    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--port", dest="port", type=int, default=7700, help="tal demo server listening port.")
    parser.add_argument("-ip", "--host", dest="host", type=str, default="0.0.0.0", help="tal demo server listening IP.")
    args = parser.parse_args()
    host = args.host
    port = args.port
    app = create_app()
    uvicorn.run(app, port=port, log_level='debug')