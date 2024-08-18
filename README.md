# SpeakingAI

**TLDR;** SpeakingAI is a demo of fully functional web AI server with audio query/answer in streaming, using LLM and RAG for backend knowledge.

**Description** 

Audio-to-audio AI has strong business use cases where users can interact with AI using voice, the audio-in, audio-out AI application can be utilized across multiple industries to enhance efficiency and user experience, such as:

- **Customer Support Automation:** Acts as a virtual assistant to handle common customer queries, reducing the need for human agents and providing 24/7 support.

- **Voice-Driven Knowledge Management:** Helps employees quickly access company information through voice queries, improving productivity in corporate environments.

- **Healthcare Assistance:** Assists healthcare professionals by retrieving medical knowledge or patient data via voice, enabling more efficient and attentive patient care.

- **Educational Support and Tutoring:** Functions as a voice-enabled tutor, offering personalized, on-demand explanations and resources to students.

- **Interactive Voice-Activated Devices:** Enhances smart home devices with a sophisticated, knowledge-rich interaction model, improving user satisfaction.

- **Market Research and Insights Gathering:** Streamlines data collection in market research by conducting voice-driven surveys and providing real-time insights.

- **Voice-Activated Training Modules:** Improves corporate training programs by enabling interactive, voice-driven learning, leading to better employee performance.

With the advance of multi-modal models such as GPT-4o, interacting with AI using voice or video is becoming a reality. You can build such an AI too, using all open source models where you can select the best suiting models and even fine-tune them according to your needs. This is an overall architecture:

The architecture is a classic LLM + RAG form where the audio components (ASR, TTS) are added to it. The following explains this classic workflow:

- **audio**: client application (such as metahuman/talking head, or pure voice) sends the audio stream to the backend AI server

- **ASR**: the ASR model transcribes the audio to text as query

- **query embedding**: the embedding model takes the query and output the embedding vector for the query

- **vector database query**: vector database returns the results (containing text meta information) according to the query

- **reranker**: the rerank model re-ranks the multiple results returned by the vector database; the reranking greatly helps to sort out irrelavant information returned by the vector search

- **LLM**: LLM model generates the answer to the query using re-ranked relevant background information

- **TTS**: TTS model output the audio wav of the text answer of LLM, and this audio stream is sent back to client

# Installation
Prepare virtual environment (use your favorite virtual management tool), for example using conda:
```bash
conda create -n speakingai python
conda activate speakingai
```

Git pull:
```bash
git clone https://github.com/awtestergit/SpeakingAI.git
cd SpeakingAI
pip install -e .
```

Create vector database for demo:
```bash
python tal_kw_populate.py
```

Start the demo server:
```bash
python tal_demo_server.py
```
Start demo client:
```bash
python tal_demo_client.py
```

In your web browser
http://127.0.0.1:8881/webui/
click 'record' to record your voice query, and click 'Stop' to send the voice query to server.

# Notes
**all the model files are pre-downloaded to the local machine**

**LLM** model is LLama3 instr 8B, but you can choose any of your favorite LLM model. <br>
Implementation code is models/llm_hf.py. Each LLM model generation-stop is different and I set the 'stop = ['<|eot_id|>']' for Llama, and sometimes I set the stop as '['<|eot_id|>', '\n\n']' to make sure the output stops at double break lines, for example.

**TTS** is Microsoft SpeechT5TTS, which is not great but quite fast. <br>
Implementation code is in models/ttsmodel.py, which also include Bark TTS. Bark is better but also slower, and you can also choose other TTS of your choose.

**Reranker** is BAAI/bge-reranker-v2-m3

**Embedding** is intfloat/multilingual-e5-large

**ASR** is Whisper

**Vector database** is Qdrant, and for the demo purpose, only adding knowledge to the vector database is implemented. For the backend knowledge, there are two areas where it can be improved:
1. a full functional vector database management service, which is to manage the vector database itself, such as add/delete/update vector database
2. add knowledge graph - using knowledge graph with vector database is an active research topic that can improve knowledge search greatly


