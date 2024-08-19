# -*- coding: utf-8 -*-
"""
from interface.interface_model import ILanguageModel
from llama_cpp import Llama, ChatCompletionRequestUserMessage, ChatCompletionRequestSystemMessage, ChatCompletionRequestAssistantMessage

class LlamaCppModel(ILanguageModel):
    def __init__(self, model_path:str, max_context_length:int, token_ex:float, seed=-1, chatformat='chatml', verbose=False) -> None:
        super().__init__(max_context_length=max_context_length, token_ex=token_ex, seed=seed)
        self.path = model_path
        self.stop = [] # stop tokens
        self.chatformat = chatformat
        self.verbose = verbose
        self.__initialize__()

    def __initialize__(self):
        self.device = self.__get_device__()
        n_ctx = int(self.MAX_LENGTH*1.1)
        self.model = Llama(model_path=self.path, n_gpu_layers=-1, n_ctx=n_ctx, seed=self.seed, chat_format=self.chatformat, verbose=self.verbose)#, n_ctx=0)
    
    def __construct_chat_message__(self, inputs, system_prompt='', assistant_prompt='', history=[]):
        messages = []
        if len(system_prompt)>0:
            prompt = ChatCompletionRequestSystemMessage()
            prompt['role'] = "system"
            prompt['content'] = system_prompt
            messages.append(prompt)
        
        if len(history)>0:
            for user, bot in history:
                prompt = ChatCompletionRequestUserMessage()
                prompt['role'] = "user"
                prompt['content'] = user
                messages.append(prompt)
                prompt = ChatCompletionRequestAssistantMessage()
                prompt['role'] = "assistant"
                prompt['content'] = bot
                messages.append(prompt)
        
        prompt = ChatCompletionRequestUserMessage()
        prompt['role'] = "user"
        prompt['content'] = inputs
        messages.append(prompt)

        if len(assistant_prompt)>0:
            prompt = ChatCompletionRequestAssistantMessage()
            prompt['role'] = "assistant"
            prompt['content'] = assistant_prompt
            messages.append(prompt)
        
        return messages
    
    def stream_chat(self, inputs, system_prompt='', assistant_prompt='', history=[], splitter='', stop=[], replace_stop=False, output_delta=False, string_outtype=False, **kwargs):
        """{'id': 'chatcmpl-a51ab02b-e02a-4be7-a265-a2b98935e3e1',
        'model': '/Users/albert/projects/llama2/llama.bak/models/llama-2-7b-chat/ggml-model-q8_0.gguf',
        'created': 1702460337,
        'object': 'chat.completion.chunk',
        'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
        """
        messages = self.__construct_chat_message__(inputs=inputs,system_prompt=system_prompt,assistant_prompt=assistant_prompt,history=history)

        key = 'max_new_tokens'
        max_tokens = -1
        if key in kwargs.keys():
            max_tokens = kwargs.pop(key)
        key = 'max_tokens' # continue to check max_tokens for different bindings
        if key in kwargs.keys():
            max_tokens = kwargs.pop(key)
        key = 'repetition_penalty'
        repeat_penalty = 1.1
        if key in kwargs.keys():
            repeat_penalty = kwargs.pop(key)
        key = 'repeat_penalty' # continue to check 
        if key in kwargs.keys():
            repeat_penalty = kwargs.pop(key)
        key = 'do_sample' # remove do_sample key
        if key in kwargs.keys():
            kwargs.pop(key)

        if len(stop) == 0: # use self stop in stream_chat
            stop = self.stop
        
        seed = self.seed if self.seed > 0 else None
        self.model.reset()
        response = self.model.create_chat_completion(messages, seed=seed, max_tokens=max_tokens, repeat_penalty=repeat_penalty, stop=stop, stream=True, **kwargs)
        answer_str = ''
        for r in response:
            key = 'content'
            delta = r['choices'][0]['delta']
            if key in delta.keys():
                if output_delta:
                    answer_str = delta[key]
                else:
                    answer_str += delta[key]
                output = answer_str if string_outtype else (answer_str, '')
                yield output# for aigc server's seek answer

class Llama3Cpp(LlamaCppModel):
    def __init__(self, model_path: str, max_context_length:int=8192, token_ex:float=0.7, seed=20240819, verbose=False, chatformat='llama-3') -> None:
        max_context_length = 5000
        token_ex = token_ex # 0.7 token ~= 1 chn words
        super().__init__(model_path, max_context_length, token_ex, seed, chatformat=chatformat, verbose=verbose)
        self.stop = ['<|eot_id|>']

    def stream_chat(self, inputs, system_prompt='', assistant_prompt='', history=[], splitter='', stop=None, replace_stop=False,  string_outtype=False, output_delta=False, **kwargs):
        stop = stop if stop is not None else []
        if not replace_stop:
            stop.extend(self.stop)
        ### default history, seems this binder need to have history as a kick start
        if (len(history)) == 0:
            history = [('Hi','Hello!')]
        return super().stream_chat(inputs, system_prompt, assistant_prompt, history, splitter, stop, string_outtype=string_outtype, output_delta=output_delta, **kwargs)
"""
