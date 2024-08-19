# SpeakingAI

**TLDR;** SpeakingAI is a demo of fully functional web AI server with audio query/answer in streaming, using LLM and RAG for backend knowledge.

## Index

* [Description](#description)
* [Architecture](#architecture)
* [Installation](#installation)
* [Notes & Model downloads](#notes)

## Update
Using Llama gguf model format is supported so that you can run the server if your GPU RAM is smaller. see [Special instruction of using llama3 GGUF](#llama-gguf)

## Description

Audio-to-audio AI has strong business use cases where users can interact with AI using voice, the audio-in, audio-out AI application can be utilized across multiple industries to enhance efficiency and user experience, such as:

- **Customer Support Automation:** Acts as a virtual assistant to handle common customer queries, reducing the need for human agents and providing 24/7 support.

- **Voice-Driven Knowledge Management:** Helps employees quickly access company information through voice queries, improving productivity in corporate environments.

- **Healthcare Assistance:** Assists healthcare professionals by retrieving medical knowledge or patient data via voice, enabling more efficient and attentive patient care.

- **Educational Support and Tutoring:** Functions as a voice-enabled tutor, offering personalized, on-demand explanations and resources to students.

- **Interactive Voice-Activated Devices:** Enhances smart home devices with a sophisticated, knowledge-rich interaction model, improving user satisfaction.

- **Market Research and Insights Gathering:** Streamlines data collection in market research by conducting voice-driven surveys and providing real-time insights.

- **Voice-Activated Training Modules:** Improves corporate training programs by enabling interactive, voice-driven learning, leading to better employee performance.

With the advance of multi-modal models such as GPT-4o, interacting with AI using voice or video is becoming a reality. You can build such an AI too, using all open source models where you can select the best suiting models and even fine-tune them according to your needs. This is an overall architecture:

## Architecture
<p align='center'>
  <img width="948" alt="SpeakingAI RAG Architecture" src="https://github.com/user-attachments/assets/cd8b50a4-7921-4556-b9d6-70b1ff761ad3">
</p>

The architecture is a classic LLM + RAG form where the audio components (ASR, TTS) are added to it. The following explains this classic workflow:

- **audio**: client application (such as metahuman/talking head, or pure voice) sends the audio stream to the backend AI server

- **ASR**: the ASR model transcribes the audio to text as query

- **query embedding**: the embedding model takes the query and output the embedding vector for the query

- **vector database query**: vector database returns the results (containing text meta information) according to the query

- **reranker**: the rerank model re-ranks the multiple results returned by the vector database; the reranking greatly helps to sort out irrelavant information returned by the vector search

- **LLM**: LLM model generates the answer to the query using re-ranked relevant background information

- **TTS**: TTS model output the audio wav of the text answer of LLM, and this audio stream is sent back to client

## Installation
Prepare virtual environment (use your favorite virtual management tool), for example using conda:
```bash
conda create -n speakingai python
conda activate speakingai
```

Git pull:
```bash
git clone https://github.com/awtestergit/SpeakingAI.git
cd SpeakingAI
pip install -r requirements.txt
```

Download all models to your local machine to speed up the first time run: <br>
Llama3 Instr 8B, BAAI/bge-reranker-v2-m3, intfloat/multilingual-e5-large, whisper-large-v3 <br>
*For model downloads, please see [Notes](#notes)

Install Qdrant docker
```bash
docker pull qdrant/qdrant
```
then
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

Create vector database for demo:
```bash
python tal_kw_populate.py
```
Config the demo server:

* modify 'config.json' according to your local pathes <br>
-- **note:** If you do not use Bark TTS, you do not need to config "TTS" and/or "SPEAKER" for Bark. The demo server uses the Microsoft SpeechT5 TTS by default. If you want to change it to Bark, you can set your Bark model path in the config.json, and uncomment the Bark model code in tal_demo_server.py (also comment out this line: tts = Speecht5TTS())

Start the demo server:
```bash
python tal_demo_server.py
```
Start demo client:
```bash
python tal_demo_client.py
```

In your web browser, go to http://127.0.0.1:8881/webui/ <br>
Click 'record' to record your voice query, and click 'Stop' to send the voice query to server. <br>
Since the knowledge in the vector database is about US economy, you can ask questions like "What is the US economy forecast?", or you can ask "What is your name", "What time is it" just for fun. <br>

**note:** the gradio client takes a few seconds to upload your voice (you can monitor the print outs of the demo client), and I am not sure why the latest Gradio behaves like this - it used to be quite fast to upload the voice audio. Once the audio is transmitted to the demo server, the server is quite fast to answer the audio back if you use Microsoft SpeechT5 TTS, which is the default, but this TTS does not pronunce numbers and you can hear that all numbers are not pronunced. Bark is a lot better but is slow.

## Notes
**all the model files are pre-downloaded to the local machine, otherwise the first time of starting the server takes quite a long time**

**LLM** model is LLama3 instr 8B, but you can choose any of your favorite LLM model. <br>
Implementation code is models/llm_hf.py. Each LLM model generation-stop is different and I set the 'stop = ['<|eot_id|>']' for Llama, and sometimes I set the stop as '['<|eot_id|>', '\n\n']' to make sure the output stops at double break lines, for example. <br>
https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

**TTS** is Microsoft SpeechT5TTS, which is not great but quite fast. <br>
Implementation code is in models/ttsmodel.py, which also include Bark TTS. Bark is better but also slower, and you can also choose other TTS of your choose. <br>
https://huggingface.co/microsoft/speecht5_tts 

or if you use Bark TTS: <br>
https://huggingface.co/suno/bark-small

**Reranker** is BAAI/bge-reranker-v2-m3
https://huggingface.co/BAAI/bge-reranker-v2-m3

**Embedding** is intfloat/multilingual-e5-large
https://huggingface.co/intfloat/multilingual-e5-large

**ASR** is Whisper
https://huggingface.co/openai/whisper-large-v3

**Vector database** is Qdrant, and for the demo purpose, only adding knowledge to the vector database is implemented. For the backend knowledge, there are two areas where it can be improved:
1. a full functional vector database management service, which is to manage the vector database itself, such as add/delete/update vector database
2. add knowledge graph - using knowledge graph with vector database is an active research topic that can improve knowledge search greatly

## Llama GGUF
If you have small GPU, or just want the model runs fast, you can use GGUF format powered by Llama.cpp. <br>

check out https://github.com/ggerganov/llama.cpp for more details if you want to quantize LLama3 8B by yourself, or you can download a ready to use model at huggingface, for example: <br>
https://huggingface.co/chatpdflocal/llama3.1-8b-gguf <br>

1. download the gguf model (or quantize it by yourself following the steps in llama.cpp github)
2. install llama cpp Python binder, https://llama-cpp-python.readthedocs.io/en/latest/
3. Note: if you use Apple Metal mac, the installation is 'CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python' to enable metal support; otherwise 'pip install llama-cpp-python'
4. in the config.json, set the LLM to your local gguf full path
5. uncomment the whole file of 'llm_llamacpp.py', and uncomment '#from models.llm_llamacpp import Llama3Cpp', '#llm = Llama3Cpp(llm_path)' in the tal_demo_server.py (comment out 'llm = Llama3(llm_path)')
6. that's it!
