import glob
import os

paths = glob.glob("/opt/DNS/DNS-Challenge/corpus_multimic/noisy/*.txt",recursive = True)

f = open("dataset_wavs_DNSMultiMic.txt", "w")

for path in paths:
  f.write(path+"/\n")

f.close()
