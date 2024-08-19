# -*- coding: utf-8 -*-
#
# @author awtestergit
# @description transformer model
#

import torch
import random
import numpy as np

from transformers import TextIteratorStreamer
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from interface.interface_model import ILanguageModel

class TransformerModels(ILanguageModel):
    class TMStoppingProcessor(LogitsProcessor):
        """
        Modified from Qwen model code
        :class:`transformers.LogitsProcessor` that enforces that when specified sequences appear, stop geration.
        enforcement is done by assign score of eos_token_id a high value

        Args:
            stop_words_ids (:obj:`List[List[int]]`):
                List of list of token ids of stop ids. In order to get the tokens of the words
                that should not appear in the generated text, use :obj:`tokenizer(bad_word,
                add_prefix_space=True).input_ids`.
            eos_token_id (:obj:`list`):
                The id of the `end-of-sequence` token or a list.
        """

        def __init__(self, stop_words_ids: list, eos_token_id: int|list[int]):

            if not isinstance(stop_words_ids, list) or len(stop_words_ids) == 0:
                raise ValueError(
                    f"`stop_words_ids` has to be a non-emtpy list, but is {stop_words_ids}."
                )
            if any(not isinstance(bad_word_ids, list) for bad_word_ids in stop_words_ids):
                raise ValueError(
                    f"`stop_words_ids` has to be a list of lists, but is {stop_words_ids}."
                )
            if any(
                any(
                    (not isinstance(token_id, (int, int)) or token_id < 0)
                    for token_id in stop_word_ids
                )
                for stop_word_ids in stop_words_ids
            ):
                raise ValueError(
                    f"Each list in `stop_words_ids` has to be a list of positive integers, but is {stop_words_ids}."
                )

            self.stop_words_ids = list(
                filter(
                    lambda bad_token_seq: bad_token_seq != [eos_token_id], stop_words_ids
                )
            )
            self.eos_token_id = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
            for stop_token_seq in self.stop_words_ids:
                assert (
                    len(stop_token_seq) > 0
                ), "Stop words token sequences {} cannot have an empty list".format(
                    stop_words_ids
                )

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
        ) -> torch.FloatTensor:
            """
            input_ids: shape as [batch, seq_length]
            scores: shape as [batch, config.vocab_size]
            """
            stopped_samples = self._calc_stopped_samples(input_ids)
            for i, should_stop in enumerate(stopped_samples):
                if should_stop:
                    for eos_token_id in self.eos_token_id:
                        scores[i, eos_token_id] = float(2**15) # if any stop found, assign high score of eos token id
            return scores

        def _tokens_match(self, prev_tokens: torch.LongTensor, bad_tokens) -> bool:
            if len(bad_tokens) == 0:
                # if bad word tokens is just one token always ban it
                return True
            elif len(bad_tokens) > len(prev_tokens):
                # if bad word tokens are longer then prev input_ids they can't be equal
                return False
            elif prev_tokens[-len(bad_tokens) :].tolist() == bad_tokens:
                # if tokens match
                return True
            else:
                return False

        def _calc_stopped_samples(self, prev_input_ids):
            stopped_samples = []
            for prev_input_ids_slice in prev_input_ids:
                match = False
                for stop_token_seq in self.stop_words_ids:
                    if self._tokens_match(prev_input_ids_slice, stop_token_seq):
                        # if tokens do not match continue
                        match = True
                        break
                stopped_samples.append(match)

            return stopped_samples

    def __init__(self, path, max_length, token_ex, seed=20240815, load_in_8bit = False, stop = ['</s>','[/INST]'], torch_dtype=torch.float16, set_pad_token=False) -> None:
        super().__init__(max_context_length=max_length, token_ex=token_ex, seed=seed)
        self.path = path
        self.load_in_8bit = load_in_8bit
        self.torch_dtype = torch_dtype
        self.stop = stop # default, should be replaced by different model
        self.set_pad_token = set_pad_token # if True, will use eos token to set pad token if model tokenizer does not have padding token
        self.__initialize__()

    def __initialize__(self):
        # set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.device = self.__get_device__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, use_fast=False)
        if self.set_pad_token:
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.path, load_in_8bit=self.load_in_8bit, torch_dtype=torch.float16, device_map='auto', local_files_only=True)
        self.model.generation_config = GenerationConfig.from_pretrained(self.path)
        self.model.eval()        

    def __generation_preprocessor__(self, stop=[], **kwargs):
        """
        stop: a list of stop strings
        """
        if len(stop)==0:
            return None
        
        generation_config = self.model.generation_config
        logits_processor = kwargs.pop("logits_processor", None)

        #stop_words_ids = self.tokenizer(stop, return_tensors='pt', padding=True, truncation=True).to(self.device)#return list[int], not tensor, return_tensors='pt').to(self.device)
        stop_words_ids = self.tokenizer(stop, padding=True, truncation=True)
        stop_words_ids = list(stop_words_ids.input_ids) if stop_words_ids else None
        #stop_words_ids = [stop_words_ids.input_ids.squeeze().tolist()] if stop_words_ids else None
        if (len(stop_words_ids)==1) and type(stop_words_ids[0]) is not list: # make it a list of list
            stop_words_ids = [stop_words_ids]

        if stop_words_ids is None and generation_config is not None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)

        if stop_words_ids is not None:
            stop_words_logits_processor = TransformerModels.TMStoppingProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)
        return logits_processor

    @classmethod
    def __construct_chat_message__(cls, inputs:str, system_prompt:str, assistant_prompt:str, history:list)->list:
        # 1. construct system prompt
        # 2. construct history
        # 3. construct input
        # 4. construct assistant prompt, ignored
        messages = []
        if len(system_prompt)>0:
            system = {
                'role': 'system',
                'content': system_prompt
            }
            messages.append(system)
        if len(history)>0:
            for user, bot in history:
                user = {
                    'role': 'user',
                    'content': user
                }
                messages.append(user)
                bot = {
                    'role': 'assistant',
                    'content': bot
                }
                messages.append(bot)
        user = {
            'role': 'user',
            'content': inputs
        }
        messages.append(user)
        return messages

    def stream_chat(self, inputs, system_prompt='', assistant_prompt='', history=[], splitter='', stop=None, replace_stop=True, output_delta=False, string_outtype=False, **kwargs):
        stop = stop if stop is not None else []
        if not replace_stop:
            stop.extend(self.stop)
        # system_prompt='', assistant_prompt=''
        # consruct message
        messages = TransformerModels.__construct_chat_message__(inputs, system_prompt, assistant_prompt, history)
        #if self.tokenizer.pad_token_id is None:
        #    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        #_inputs = self.tokenizer.decode(tokenized_chat[0]) # the whole messages
        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.model.eval()
        logits_processor = self.__generation_preprocessor__(stop, **kwargs)
        self.model.generate(tokenized_chat, streamer=streamer, logits_processor=logits_processor, **kwargs)
        texts = ''
        # splitter = _inputs if len(splitter) == 0 else splitter
        for t in streamer:
            if output_delta: # output delta only
                texts = t
            else:
                texts += t
                texts = texts.split(splitter)[-1] if len(splitter)>0 else texts # split by splitter, if any
            yield texts

class Llama3(TransformerModels):
    def __init__(self, path, seed=20240815, load_in_8bit = False) -> None:
        max_length = 5000 # seq length 8k
        token_ex = 0.7 #
        stop = ['<|eot_id|>']
        torch_dtype = torch.bfloat16
        super().__init__(path, max_length, token_ex, seed=seed, load_in_8bit=load_in_8bit, stop=stop, torch_dtype=torch_dtype, set_pad_token=True)

    def __initialize__(self):
        seq_length = 8000 #model max seq length
        # set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.device = self.__get_device__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, model_max_length=seq_length, use_fast=False)
        if self.set_pad_token:
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.path, load_in_8bit=self.load_in_8bit, torch_dtype=torch.float16, device_map='auto', local_files_only=True)
        self.model.generation_config = GenerationConfig.from_pretrained(self.path)
        self.model.eval()