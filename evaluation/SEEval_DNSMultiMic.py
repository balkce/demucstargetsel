# also requires: pip install denoiser
import torch
import torchaudio
import math
import mir_eval
import time
import os
import sys
import glob
import psutil
import shutil
import speechbrain as sb
from denoiser import pretrained
from denoiser.dsp import convert_audio
import inspect
from torch.nn import functional as F

from speechbrain.pretrained import EncoderClassifier
embedding_extractor = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

class SEEval():
  dir_path=""
  results_dict = {
    "wav_id": "",
    "babble_snr": 0.0,
    "babble_num": 0,
    "noise_snr": 0.0,
    "reverb_scale": 0.0,
    "window_size": 0,
    "response_time": 0.0,
    "SIR": 0.0
  }
  curr_model = {
    "id": 0,
    "type": "",
    "model": 0
  }
  
  
  def combine_embed (self, embed,signal):
      embed_lens = (torch.ones(embed.shape[0],1,1)*embed.shape[2]).to(signal.device)
      return torch.cat((embed_lens,embed,signal),2)
  
  def combine_interf (self, signal,interf):
      return torch.cat((signal,interf),2)
  
  def deserialize_model(self, package, strict=False):
      """deserialize_model.

      """
      klass = package['class']
      if self.curr_model["type"] == "embed":
          import denoiserembed.demucs
          klass = denoiserembed.demucs.Demucs
      elif self.curr_model["type"] == "phase":
          import denoiserphase.demucs
          klass = denoiserphase.demucs.Demucs
      kwargs = package['kwargs']
      if strict:
          model = klass(*package['args'], **kwargs)
      else:
          sig = inspect.signature(klass)
          kw = package['kwargs']
          for key in list(kw):
              if key not in sig.parameters:
                  del kw[key]
          model = klass(*package['args'], **kw)
      model.load_state_dict(package['state'])
      return model
  
  def load_demucsmodel(self, model_path):
    pkg = torch.load(model_path, 'cpu')
    if 'model' in pkg:
        if 'best_state' in pkg:
            pkg['model']['state'] = pkg['best_state']
        model = self.deserialize_model(pkg['model'])
    else:
        model = self.deserialize_model(pkg)
    return model
  
  def __init__(self, path=""):
    self.dir_path = path
  
  def load_model(self, model_number):
    self.curr_model["id"] = model_number
    if model_number == 0:
      self.curr_model["type"] = "original"
      self.curr_model["model"] = pretrained.dns64()
    elif model_number == 1:
      self.curr_model["type"] = "embed"
      self.curr_model["model"] = self.load_demucsmodel("./pretrained_models/demucsembed/best.th")
    elif model_number == 2:
      self.curr_model["type"] = "phase"
      self.curr_model["model"] = self.load_demucsmodel("./pretrained_models/demucsphase/best.th")
    else:
      print("Invalid model number (0...2).")

  def run_model(self, txt_path, window_size):
    dir_path_elems = txt_path.split("/")
    
    while("" in dir_path_elems):
      dir_path_elems.remove("")
    
    txt_filename = dir_path_elems[-1]
    txt_filebasename = txt_filename.split(".txt")[0]
    
    txt_info = txt_filebasename.split("_")
    
    self.results_dict["wav_id"] = txt_info[-1]
    self.results_dict["noise_snr"] = float(txt_info[0].split("snr")[-1])
    self.results_dict["reverb_scale"] = float(txt_info[1].split("rt")[-1])
    self.results_dict["babble_num"] = int(txt_info[2].split("ints")[-1])
    self.results_dict["babble_snr"] = float(txt_info[3].split("sir")[-1])
    self.results_dict["window_size"] = window_size
    #print(self.results_dict)
    
    int_paths = open(txt_path).readlines()
    
    wav_path = ""
    if self.curr_model["type"] == "original":
        wav_path = str(int_paths[3].rstrip())
    elif self.curr_model["type"] == "embed":
        wav_path = str(int_paths[3].rstrip())
    elif self.curr_model["type"] == "phase":
        wav_path = txt_path.replace(".txt",".wav")
    else:
        wav_path = ""
    
    #print(wav_path)
    
    local_wavs = glob.glob("./*.wav")
    for local_wav in local_wavs:
      os.remove(local_wav)
    
    noisy, sr = torchaudio.load(wav_path)
    
    result_len = noisy.size(1)
    if window_size == 0:
      one_window = True
      window_size = result_len
      window_num = 1
    else:
      one_window = False
      window_num = math.floor(result_len/window_size)
      if result_len % window_size > 0:
        window_num += 1
    
    #creating original signals
    references = torch.zeros(1,result_len)
    estimations = torch.zeros(1,result_len)
    clean = sb.dataio.dataio.read_audio(str(int_paths[0].rstrip()))
    if clean.size(0) < result_len: #shouldn't happen, but c'est la vie
      clean = torch.cat((clean.unsqueeze(0),torch.zeros(1,result_len-clean.size(0))),1).squeeze(0)
    
    references[0,:] = clean[:result_len].detach()
    
    
    win_result = torch.zeros(1,window_num*window_size)
    win_result_len = win_result.size(1)
    noisy_win = torch.zeros(1,window_size)
    exec_time_mean = 0.0
    exec_time_i = 0
    
    if self.curr_model["type"] == "embed":
      embed_path = str(int_paths[1].rstrip())
      embed_signal, sr = torchaudio.load(embed_path)
      embedding = embedding_extractor.encode_batch(embed_signal)
    
    for i in range(0,window_num):
      if one_window:
        noisy_win = noisy
      else:
        if i == window_num-1:
          noisy_win = torch.cat((noisy[:,(window_size*i):],torch.zeros(1,win_result_len-result_len)),1)
        else:
          noisy_win = noisy[:,(window_size*i):(window_size*(i+1))]
      
      if self.curr_model["type"] == "embed":
        noisy_win = noisy_win.unsqueeze(0)
        noisy_win = self.combine_embed(embedding,noisy_win)
      elif self.curr_model["type"] == "phase":
        noisy_win = noisy_win.unsqueeze(0)
        interf_path = str(int_paths[2].rstrip())
        interf_signal, sr = torchaudio.load(interf_path)
        interf_signal = interf_signal.unsqueeze(0)
        noisy_win = self.combine_interf (noisy_win,interf_signal)
        
      start_time = time.time()
      #wav = convert_audio(noisy_win, 16000, model.sample_rate, model.chin)
      with torch.no_grad():
        model_result = self.curr_model["model"](noisy_win)[0]
      exec_time = time.time() - start_time
      win_result[:,(window_size*i):(window_size*(i+1))] = model_result
      
      exec_time_mean += exec_time
      exec_time_i += 1
    
    exec_time_mean /= exec_time_i
    
    estimations[0,:] = win_result[:,:result_len].squeeze(0).detach()
    
    #sb.dataio.dataio.write_audio("seeval-clean.wav",references[0,:],16000)
    #sb.dataio.dataio.write_audio("seeval-result.wav",estimations[0,:],16000)
    #sb.dataio.dataio.write_audio("seeval-noisy.wav",noisy[0,:],16000)
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(references.numpy(), estimations.numpy())
    self.results_dict["SIR"] = sdr[0].item()
    
    self.results_dict["response_time"] = exec_time_mean
    return self.results_dict

if (len(sys.argv) < 2):
  print("Required: directory where TXT files reside.")
  exit()

corpus_path = sys.argv[1]

if not os.path.exists(corpus_path):
  print("Does not exists: "+corpus_path)
  exit()

#eval_window_size = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 0]
eval_window_size = [0]

paths = glob.glob(os.path.join(corpus_path,"*.txt"))
paths.sort()
paths_num = len(paths)

if os.path.exists("curr_dataset_wav.txt"):
  curr_path_file = open("curr_dataset_wav.txt","r")
  path_i_start = int(curr_path_file.read())
  curr_path_file.close()
  
  curr_model_file = open("curr_model.txt","r")
  model_i_start = int(curr_model_file.read())
  curr_model_file.close()
  
  curr_len_file = open("curr_len.txt","r")
  win_i_start = int(curr_len_file.read())
  curr_len_file.close()
else:
  curr_path_file = open("curr_dataset_wav.txt","w")
  curr_path_file.write(str(0))
  curr_path_file.close()
  path_i_start = 0
  
  curr_model_file = open("curr_model.txt","w")
  curr_model_file.write(str(0))
  curr_model_file.close()
  model_i_start = 0
  
  curr_len_file = open("curr_len.txt","w")
  curr_len_file.write(str(0))
  curr_len_file.close()
  win_i_start = 0
  
  eval_results_file = open("curr_results.csv","w")
  eval_results_file.write("model,wav_id,reverb_scale,babble_num,babble_snr,noise_snr,window_size,response_time,SIR,mem\n")
  eval_results_file.close()

seeval = SEEval(".")

this_process = psutil.Process(os.getpid())

window_size = eval_window_size[win_i_start];

print("Evaluating model "+str(model_i_start)+" with window size: "+str(window_size))
seeval.load_model(model_i_start)
ini_mem = this_process.memory_info().rss

for path_i in range(path_i_start,paths_num):
  curr_path_file = open("curr_dataset_wav.txt","w")
  curr_path_file.write(str(path_i))
  curr_path_file.close()
  
  path = paths[path_i].rstrip()
  
  print("  ->",str(path_i),":",path)
  eval_result = seeval.run_model(path, window_size)
  #print(eval_result)
  
  curr_mem = this_process.memory_info().rss
  
  if curr_mem >= ini_mem*10:
    print("Memory grew too much, from",str(ini_mem),"bytes to",str(curr_mem),"bytes. resetting.")
    exit()
  
  eval_results_file = open("curr_results.csv","a")
  eval_results_file.write(str(model_i_start)+","+eval_result["wav_id"]+","+str(eval_result["reverb_scale"])+","+str(eval_result["babble_num"])+","+str(eval_result["babble_snr"])+","+str(eval_result["noise_snr"])+","+str(eval_result["window_size"])+","+str(eval_result["response_time"])+","+str(eval_result["SIR"])+","+str(curr_mem)+"\n")
  eval_results_file.close()

curr_path_file = open("curr_dataset_wav.txt","w")
curr_path_file.write(str(0))
curr_path_file.close()

if win_i_start < len(eval_window_size)-1:
  curr_len_file = open("curr_len.txt","w")
  curr_len_file.write(str(win_i_start+1))
  curr_len_file.close()
else:
  curr_len_file = open("curr_len.txt","w")
  curr_len_file.write(str(0))
  curr_len_file.close()
  
  curr_model_file = open("curr_model.txt","w")
  curr_model_file.write(str(model_i_start+1))
  curr_model_file.close()
  
  shutil.copyfile("curr_results.csv", "curr_results-model_"+str(model_i_start)+".csv")
  
  eval_results_file = open("curr_results.csv","w")
  eval_results_file.write("model,wav_id,reverb_scale,babble_num,babble_snr,noise_snr,window_size,response_time,SIR,mem\n")
  eval_results_file.close()


