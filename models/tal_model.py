# -*- coding: utf-8 -*-
#
# @author awtestergit
# @description TTS, ASR and LLM, for digital human applications, using parrots TTS & ASR
#
import numpy as np
from typing import Iterator
import datetime
from interface.interface_model import ILanguageModel, IASRModel, ITTSModel, IRerankerModel
from interface.interface_tal import ITALModel, TALAudioRequest, tal_encoder
from qdrantclient_vdb.qdrant_knowledge import KnowledgeWarehouse
import traceback

class TALModel(ITALModel):
    def __init__(self, llm: ILanguageModel, reranker:IRerankerModel, asr: IASRModel, tts: ITTSModel, kw_vdb:KnowledgeWarehouse, ai_name="AI", vdb_conf=0.6) -> None:
        super().__init__(llm, reranker, asr, tts)
        #LLM
        self.name = ai_name
        #knowledge warehouse
        self.kw_vdb = kw_vdb
        # confidence
        self.vdb_conf = vdb_conf

    def audio_in(self, audio: bytes|np.ndarray, rate:int=0, bit_depth:int=16, target_sample_rate=16000) -> str:
        text = self.asr.voice_to_text(audio, rate, bit_depth=bit_depth, target_sample_rate=target_sample_rate)
        return text
    
    def audio_out(self, inputs: str, sample_rate:int) -> tuple[int, np.ndarray]|None:
        """
        input: str
        outputs: (sample_rate, audio array)
        """
        data =  self.tts.text_to_data(inputs, sample_rate)
        return data
    
    def query(self, query: str, history=[], splitter='', stop=None, replace_stop=False, top=3, conf=0.9, rerank_min_score=0.1, **kwargs) -> tuple[Iterator, str]:
        """
        inputs: 
            query: the ask
            history: chat history if provided
            splitter: split the LLM answer, the last one, -1, will be returned as answer
            stop: stop words
            replace_stop, if True, will replace LLM's own stop words, else, will append to the internal stop words
            top: the top number of answers with match conf >= conf, default top 1
            conf: the confidence score for the vector db search
            reranker_min_score: the minimum score of the reranker returns for a query and answer to be considerd relevant at all
        output: (answers, sources), where answers is an iterator, sources is a string
        """
        answers = None

        system_prompt_context = """You are AI, name is {name}. It is now {current}. You need to answer user's question based on the provided background information.

Background Information:
[
{context}
]

Remember: It is essential to identify key information in the background information that pertains to user's question to provide more accurate responses.
Remember: Your responses must not contradict the facts presented in the background information.
"""
        system_prompt_no_context = """You are AI, name is {name}. It is now {current}. Please answer user's question friendly."""
        # current time
        current_time = datetime.datetime.now()
        current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        current_day = datetime.datetime.today().strftime('%A')
        current_time = f"{current_time}, {current_day}"

        #query knowledge base, and consulting with LLM for an answer
        contexts = []
        try:
            contexts = self.kw_vdb.query_knowledge(query, top_k=top, threshold=conf) # the top number of results with scores > conf
        except:
            e = traceback.format_exc()
            print(e)

        # print("...contexts...")
        # print(contexts)
        # print()

        # rerank them
        score_context = []
        for ctx in contexts:
            meta = ctx['meta'] # only meta part
            score = self.reranker.rerank_score(query, meta) # rerank element against text indexed at INDEX[0] array
            score = score[1] # score returned as (True, score)
            score_context.append([score, ctx]) # save to list
        contexts = [] # empty it
        if len(score_context)>0:
            score_context.sort(key=lambda x: x[0], reverse=True) # sort by score, descending
            max_score = score_context[0][0]
            if max_score > rerank_min_score: # if rank score too low, discard
                contexts = [ctx[1] for ctx in score_context[:top] if ctx[0] > rerank_min_score]

        # print("...after reranker contexts...")
        # print(contexts)
        # print()

        # construct prompt
        if len(contexts)==0:# cannot find an answer with confidence from vdb
            sources = ["I can not find relevent contexts from knowledge warehouse, the following is a generic answer."]
            prompt = system_prompt_no_context.format(name=self.name, current=current_time)
        else: #with context
            sources = [f"""{ctx['meta']}""" for ctx in contexts]
            # context only meta
            contexts = [f"{ctx['meta']}" for ctx in contexts]
            prompt = system_prompt_context.format(name=self.name, current=current_time, context=contexts)

        # output delta only, such as in sentence "My name is AI", the iterator will be output as 'delta': my, name, is, AI
        answers = self.llm.stream_chat(inputs=query,system_prompt=prompt, history=history, splitter=splitter, stop=stop, replace_stop=replace_stop, string_outtype=True, output_delta=True, **kwargs)

        # append history, only question so far, the answer will be added at callback
        history.append((query, None))

        return answers, sources

    def tal_audio_response(self,query:str, answers: Iterator, history=[], sample_rate=16000) -> Iterator:
        """
        query: the question be asked
        answers: the answers to be streamed into audio
        history: chat history
        sample_rate: the sample rate of audio
        call audio_out, and convert to  generator of bytes
        output in byte sequence format as: 'header_size|header|wav'
        reponses:
            iterator of bytes
        """
        answer = ''
        r = TALAudioRequest()
        order = 0

        # get complete sentences from the streamed answers
        sentence_iterator = self.__llm_outout_generator__(answers)
        for sentence in sentence_iterator:
            audio = self.audio_out(sentence, sample_rate=sample_rate)
            sample_rate, dtype, audio_array = audio # int, 'float32', nd.array
            if len(audio_array)==0:
                continue
            # keep answer
            answer += sentence
            order += 1
            r.order = order
            r.sign = 1 if order==1 else 2 #start if order is 1, else 2 meaning the middle message
            r.sampleRate = sample_rate
            r.audioType = dtype
            r.answer = answer
            r.question = query
            wav = audio_array.tobytes() # nd.array to bytes
            yield tal_encoder(r, wav=wav)

        # add to history
        if len(history) > 0:
            last_item = history.pop()
            query = last_item[0]
            history.append((query, answer))

        # done, now send the last message
        r = TALAudioRequest()
        r.sign = 3 #end
        order += 1
        r.order = order
        yield tal_encoder(r) # the end


    def __llm_outout_generator__(self, inputs:Iterator):
        """
        create a complete sentence from llm outputs
        inputs: is an iterator, where stream_chat generates new texts in sequence, like: My, name, is, AI, 
            where each word (my, name etc) is the output when output_delta is set
        """
        current_start, current_end = 0,0
        end_chars = ['。','.','！','？','.','?', ',', '!', ':'] # end of sentence

        buffer = ''
        for text in inputs:
            if len(text) == 0:
                continue

            buffer = buffer + text
            # if sentence
            end_char = text if len(text) == 1 else text.strip(' ')[-1] # strip the ending whitespace
            if end_char in end_chars:
                #print(f"....buffer is: {buffer}")
                yield buffer
                # reset buffer
                buffer = ''

        # here, need to check if buffer is not empty
        if len(buffer) > 0:
            #print(f"....buffer ends with: {buffer}")
            yield buffer
