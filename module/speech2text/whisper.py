#!/usr/bin/env python3
import os
import sys
import argparse

import torch
from transformers import pipeline
from typing import Optional, List
from fastapi import UploadFile, Form
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

import openedai
import tempfile
import asyncio
import io
from io import BytesIO
import glob 
from pyannote.audio import Pipeline
import soundfile as sf
import numpy as np

# Configure logging
import logging
logging.basicConfig(level=logging.DEBUG)  # Log all messages with level DEBUG and higher
log = logging.getLogger(__name__)


pipe = None
app = openedai.OpenAIStub()

def remove_old_wav_files():
    # Path to /tmp directory (this might be different in your case if using custom temp directory)
    temp_directory = "/tmp"

    # Use glob to find all .wav files in the /tmp directory
    wav_files = glob.glob(os.path.join(temp_directory, "*.wav"))

    # Remove each .wav file found
    if wav_files:
        for wav_file in wav_files:
            os.remove(wav_file)


def diarization(file_data,file):

    remove_old_wav_files()

    file_like = BytesIO(file_data)

    data, samplerate = sf.read(file_like) 

    file_like.seek(0)
    
    # Load the VAD pipeline from pyannote
    pipeline = Pipeline.from_pretrained('/app/config.yaml')

    duration_seconds = len(data) / samplerate

    if duration_seconds > 60:
        log.info("Audio is longer than 1 minute.")
        pipeline.to(torch.device("xpu"))
    
    # Apply the pipeline to get diarization (including speech segments)
    diarization = pipeline({'audio': file_like, 'uri': file.filename})

    first_speaker = None
    speech_segments = []

    # Iterate through the diarization
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Set the first speaker based on the first turn
        if first_speaker is None:
            first_speaker = speaker
            log.info(f"First speaker detected: {first_speaker}")

        # Now check if the current speaker is the first speaker
        if speaker == first_speaker:
            speech_segments.append((turn.start, turn.end))
            log.info(f"Speech detected from {turn.start:.2f}s to {turn.end:.2f}s by {first_speaker}")

    output_audio = []

    for start, end in speech_segments:
        start_ms = int(start * samplerate)
        end_ms = int(end * samplerate)
        output_audio.append(data[start_ms:end_ms])

    if output_audio:
        output_audio = np.concatenate(output_audio, axis=0)
        # Save to new file
        sf.write('filtered_audio.wav', output_audio, samplerate)
        output_path='/app/filtered_audio.wav'
        return output_path
    else:
        log.info("No matching segments found!")
        return None

async def whisper(file, response_format: str, **kwargs):

    global pipe

    file_data = await file.read()  

    try:
        processed_file_path = await asyncio.to_thread(diarization, file_data,file)
        log.info(f"Diarization completed: {processed_file_path}")
    except Exception as e:
        log.error(f"Diarization failed: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


    # Ensure the file exists before reading
    while not os.path.exists(processed_file_path):
        await asyncio.sleep(0.1)

    with open(processed_file_path, "rb") as f:
        processed_data = f.read()


    result = pipe(processed_data, **kwargs)

    filename_noext, ext = os.path.splitext(file.filename)

    if response_format == "text":
        return PlainTextResponse(result["text"].strip(), headers={"Content-Disposition": f"attachment; filename={filename_noext}.txt"})

    elif response_format == "json":
        return JSONResponse(content={ 'text': result['text'].strip() }, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"})
    
    elif response_format == "verbose_json":
        chunks = result["chunks"]

        response = {
            "task": kwargs['generate_kwargs']['task'],
            #"language": "english",
            "duration": chunks[-1]['timestamp'][1],
            "text": result["text"].strip(),
        }
        if kwargs['return_timestamps'] == 'word':
            response['words'] = [{'word': chunk['text'].strip(), 'start': chunk['timestamp'][0], 'end': chunk['timestamp'][1] } for chunk in chunks ]
        else:
            response['segments'] = [{
                    "id": i,
                    #"seek": 0,
                    'start': chunk['timestamp'][0],
                    'end': chunk['timestamp'][1],
                    'text': chunk['text'].strip(),
                    #"tokens": [ ],
                    #"temperature": 0.0,
                    #"avg_logprob": -0.2860786020755768,
                    #"compression_ratio": 1.2363636493682861,
                    #"no_speech_prob": 0.00985979475080967
            } for i, chunk in enumerate(chunks) ]
        
        return JSONResponse(content=response, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}_verbose.json"})

    elif response_format == "srt":
            def srt_time(t):
                return "{:02d}:{:02d}:{:06.3f}".format(int(t//3600), int(t//60)%60, t%60).replace(".", ",")

            return PlainTextResponse("\n".join([ f"{i}\n{srt_time(chunk['timestamp'][0])} --> {srt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
                for i, chunk in enumerate(result["chunks"], 1) ]), media_type="text/srt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.srt"})

    elif response_format == "vtt":
            def vtt_time(t):
                return "{:02d}:{:06.3f}".format(int(t//60), t%60)
            
            return PlainTextResponse("\n".join(["WEBVTT\n"] + [ f"{vtt_time(chunk['timestamp'][0])} --> {vtt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
                for chunk in result["chunks"] ]), media_type="text/vtt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.vtt"})


@app.post("/v1/audio/transcriptions")
async def transcriptions(
        file: UploadFile,
        model: str = Form(...),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),
        timestamp_granularities: List[str] = Form(["segment"])
    ):
    global pipe

    try:
        kwargs = {'generate_kwargs': {'task': 'transcribe'}}

        if language:
            kwargs['generate_kwargs']["language"] = language
    # May work soon, https://github.com/huggingface/transformers/issues/27317
    #    if prompt:
    #        kwargs["initial_prompt"] = prompt
        if temperature:
            kwargs['generate_kwargs']["temperature"] = temperature
            kwargs['generate_kwargs']['do_sample'] = True

        if response_format == "verbose_json" and 'word' in timestamp_granularities:
            kwargs['return_timestamps'] = 'word'
        else:
            kwargs['return_timestamps'] = response_format in ["verbose_json", "srt", "vtt"]

        return await whisper(file, response_format, **kwargs)
  
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='whisper.py',
        description='OpenedAI Whisper API Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model', action='store', default="distil-whisper/distil-large-v3", help="The model to use for transcription. Ex. openai/whisper-large-v3")
    parser.add_argument('-d', '--device', action='store', default="auto", help="Set the torch device for the model. Ex. xpu:0")
    parser.add_argument('-t', '--dtype', action='store', default="auto", help="Set the torch data type for processing (float32, float16, bfloat16)")
    parser.add_argument('-P', '--port', action='store', default=9000, type=int, help="Server tcp port")
    parser.add_argument('-H', '--host', action='store', default='localhost', help="Host to listen on, Ex. 0.0.0.0")
    parser.add_argument('--preload', action='store_true', help="Preload model and exit.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.device == "auto":
        device = "xpu" if torch.xpu.is_available() else "cpu"

    if args.dtype == "auto":
        if torch.xpu.is_available():
            dtype = torch.bfloat16 if torch.xpu.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 if args.dtype == "float16" else torch.float32

        if dtype == torch.bfloat16 and not torch.xpu.is_bf16_supported():
            print("bfloat16 not supported on this hardware, falling back to float16", file=sys.stderr)
            dtype = torch.float16

    pipe = pipeline(
        "automatic-speech-recognition", 
        model=args.model, 
        device=device, 
        chunk_length_s=30, 
        torch_dtype=dtype
    )
    if args.preload:
        sys.exit(0)

    app.register_model('whisper-1', args.model)

    uvicorn.run(app, host=args.host, port=args.port) # , root_path=cwd, access_log=False, log_level="info", ssl_keyfile="cert.pem", ssl_certfile="cert.pem")
