import os
import subprocess

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

#eval_window_size = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 0]
eval_window_size = [0]

paths_num = file_len("dataset_wavs_DNSMultiMic.txt")

corpus_path = "/opt/DNS/DNS-Challenge/corpus_multimic/noisy"

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
  path_i_start = 0
  model_i_start = 0
  win_i_start = 0

while path_i_start < paths_num and win_i_start < len(eval_window_size) and model_i_start < 3:
  subprocess.call("python SEEval_DNSMultiMic.py "+corpus_path, shell=True)
  
  curr_path_file = open("curr_dataset_wav.txt","r")
  path_i_start = int(curr_path_file.read())
  curr_path_file.close()
  
  curr_model_file = open("curr_model.txt","r")
  model_i_start = int(curr_model_file.read())
  curr_model_file.close()
  
  curr_len_file = open("curr_len.txt","r")
  win_i_start = int(curr_len_file.read())
  curr_len_file.close()

