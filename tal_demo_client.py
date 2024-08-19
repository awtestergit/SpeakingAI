# -*- coding: utf-8 -*-

"""
    TAL Demo client
    Author: awtestergit
"""
import gradio as gr
import logging
from interface.interface_tal import TALAudioRequest, tal_decoder, tal_encoder
import pyaudio
import requests
from fastapi import FastAPI
import uvicorn
from argparse import ArgumentParser
import numpy as np

def main(
        server_addr:str,
        server_port:int,
        local_addr:str = "0.0.0.0",
        local_port:int = 8881,
):

    app=FastAPI()

    # start webui
    with gr.Blocks() as sa:
        gr.Markdown("""<h1><center> TAL Demo Client </center></h1>""")
        with gr.Row():
            with gr.Column(scale = 3):
                with gr.Row():
                    with gr.Column(visible=True) as gr_dialog:
                        dialog = gr.Chatbot(label="Chatbox", visible=True, min_width=1, height=400)
                with gr.Row(equal_height=True):
                    recording = gr.Audio(label="Click to speak", sources='microphone', type='numpy')
        # sa block load
        ### functions block ###
        def user_query(wav, dialog):
            rate, data = wav
            req = TALAudioRequest()
            req.sampleRate = rate
            if data.dtype == np.int32:
                req.bitDepth = 32 # wav from gradio is int32
                req.audioType = 'int32'
            req.channels = 1
            data = data.tobytes()

            dialog.append(['', ''])# append message
            yield dialog

            _url = f"http://{server_addr}:{server_port}/audio_in"

            print()
            print(_url)
            print()

            rd = tal_encoder(tal=req,wav=data)
            res = requests.post(_url, data=rd, stream=True)

            answer = ''
            buffer = b''
            end = False
            p = pyaudio.PyAudio()
            for r in res.iter_content(chunk_size=None):# .iter_lines():
                if end:
                    break

                buffer += r
                while (len(buffer) > 0):
                    output = tal_decoder(buffer)
                    
                    if output is None: # if only partial data received
                        buffer += output
                        break

                    (r, data), rest_bytes = output

                    sign = r['sign']
                    if sign == 3:
                        print('end iter')
                        end = True
                        break
                    order = r['order']
                    answer = r['answer']
                    question = r['question']
                    sr = r['sampleRate']
                    channels = r['channels']
                    bd = r['bitDepth']
                    at = r['audioType']

                    print(f"incoming audio, sample rate: {sr}, audioType: {at}, channels: {channels}, bit depth: {bd}, sign: {sign}, order: {order}, question: {question}, answer: {answer}")
                    print()

                    if at == 'float32':
                        audio_format = pyaudio.paFloat32
                    else:
                        sw = bd/8
                        audio_format = p.get_format_from_width(sw)
                    out_stream = p.open(rate=sr,format=audio_format,channels=channels, output=True)
                    out_stream.write(data)
                    out_stream.close()
                    
                    dialog[-1] = [question, answer]

                    yield dialog

                    buffer = rest_bytes
            p.terminate()

        ### answer/query
        # submit
        recording.stop_recording(user_query, inputs=[recording, dialog], outputs=[dialog])

    # mount gradio to fastapi app
    sa.queue()
    webui_path = "/webui"
    app = gr.mount_gradio_app(app=app, blocks=sa, path=webui_path) # http://127.0.0.1:8881/webui

    #加上这个就可以在运行main.py文件时，就运行uvicorn服务
    uvicorn.run(app=app,host=local_addr, port=local_port)

    # end
    logging.info("End...")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--port", dest="port", type=int, default=7700, help="tal demo server listening port.")
    parser.add_argument("-ip", "--host", dest="host", type=str, default="127.0.0.1", help="tal demo server listening IP.")
    args = parser.parse_args()
    host = args.host
    port = args.port
    main(server_addr=host, server_port=port)