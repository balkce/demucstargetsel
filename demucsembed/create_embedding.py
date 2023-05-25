import json
import logging
import os
import sys
import re
import torch
import torchaudio
import torchaudio.transforms as T


from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import matplotlib.pyplot as plt
import numpy as np

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    return speech_array

if (len(sys.argv) < 3):
  print("Paths missing for input WAV file and output JSON file.")
  exit()

wav_path = sys.argv[1]
json_path = sys.argv[2]

print("Loading audio data from: "+wav_path)
audio_data, audio_sr = torchaudio.load(wav_path)

target_sr = 16000
if audio_sr != target_sr:
  transform_forward = T.Resample(audio_sr, 16000)
  print("Re-sampling audio data to target sample rate ("+str(target_sr)+")")
  audio_data = transform_forward(audio_data)

print("Loading encoder model")
embedding_extractor = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

print("Creating embedding")
embedding = torch.Tensor(embedding_extractor(audio_data[None])).numpy()[0]


print("Storing embedding in: "+json_path)
embedding_dict = {'embedding': embedding.tolist()}
json_file = open(json_path,"w")
json.dump(embedding_dict, json_file, indent=4)
json_file.close()

