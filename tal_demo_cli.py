# -*- coding: utf-8 -*-

"""
    TAL Demo client using terminal console
    Author: awtestergit
    Portions of code modified from Mycroft AI Inc. open source code
"""

from interface.interface_tal import TALAudioRequest, tal_decoder, tal_encoder
import pyaudio
import numpy as np
from sys import stdin
from termios import tcsetattr, tcgetattr, TCSADRAIN
import tty
from select import select
import requests
from argparse import ArgumentParser

def main(        
        server_addr:str,
        server_port:int,
        p,
):
    print("***************************************")
    print('Press & release space to record')
    print('    - after your recording, press & release the space again to send audio to server.')
    print('Press esc to exit...')
    print("***************************************")

    RECORD_KEY = ' '
    EXIT_KEY_CODE = 27

    # helper functions
    def wait_to_continue():
        while True:
            c = stdin.read(1)
            if c == RECORD_KEY:
                return True
            elif ord(c) == EXIT_KEY_CODE:
                return False

    def record_until(p, should_return, sample_rate, channels, sample_width):
        chunk_size = 1024
        stream = p.open(format=p.get_format_from_width(sample_width), channels=channels,
                        rate=sample_rate, input=True, frames_per_buffer=chunk_size)
        frames = []
        while not should_return():
            frames.append(stream.read(chunk_size))

        stream.stop_stream()
        stream.close()

        return b''.join(frames)

    def key_pressed():
        return select([stdin], [], [], 0) == ([stdin], [], [])

    def record_until_key(p, sample_rate=16000, channels=1, sample_width=2):
        def should_return():
            return key_pressed() and stdin.read(1) == RECORD_KEY

        return record_until(p, should_return, sample_rate, channels, sample_width)
    # end helper functions

    while True:
        # if esc key
        if not wait_to_continue():
            break

        audio_data = b''
        sample_rate = 16000 # sample rate
        channels = 1
        bit_depth = 16 # sample int16
        sample_width = bit_depth / 8 

        print('Recording...')

        audio_data = record_until_key(p, sample_rate, channels, sample_width)
 
        req = TALAudioRequest()
        req.sampleRate = sample_rate
        req.channels = 1
        req.bitDepth = 16

        _url = f"http://{server_addr}:{server_port}/audio_in"

        print()
        print(f"Sending audio to {_url}...")
        print()

        rd = tal_encoder(tal=req,wav=audio_data)
        res = requests.post(_url, data=rd, stream=True)

        answer = ''
        question = ''
        buffer = b''
        end = False
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

                if at == 'float32':
                    audio_format = pyaudio.paFloat32
                else:
                    sw = bd/8
                    audio_format = p.get_format_from_width(sw)
                out_stream = p.open(rate=sr,format=audio_format,channels=channels, output=True)
                out_stream.write(data)
                out_stream.close()

                buffer = rest_bytes
        
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    # end while true
    # end

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--port", dest="port", type=int, default=7700, help="tal demo server listening port.")
    parser.add_argument("-ip", "--host", dest="host", type=str, default="127.0.0.1", help="tal demo server listening IP.")
    args = parser.parse_args()
    host = args.host
    port = args.port

    orig_settings = tcgetattr(stdin)

    def show_input():
        tcsetattr(stdin, TCSADRAIN, orig_settings)

    def hide_input():
        tty.setcbreak(stdin.fileno())

    p = pyaudio.PyAudio()

    try:
        hide_input()
        main(server_addr=host, server_port=port, p=p)    
    finally:
        tcsetattr(stdin, TCSADRAIN, orig_settings)
        p.terminate()

    