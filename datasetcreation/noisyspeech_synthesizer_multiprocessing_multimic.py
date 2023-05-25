"""
@author: chkarada
"""

# Note that this file picks the clean speech files randomly, so it does not guarantee that all
# source files will be used


import os
import glob
import argparse
import ast
import configparser as CP
from itertools import repeat
import multiprocessing
from multiprocessing import Pool
import random
from random import shuffle
import librosa
import numpy as np
from audiolib_multimic import is_clipped, audioread, audiowrite, pyroom_mixer, activitydetector
import utils
import json

PROCESSES = multiprocessing.cpu_count()
MAXTRIES = 50
MAXFILELEN = 100

np.random.seed(2)
random.seed(3)

clean_counter = None
noise_counter = None
interf_counter = None

users = []

def init(args1, args2, args3):
    ''' store the counter for later use '''
    global clean_counter, noise_counter, interf_counter
    clean_counter = args1
    noise_counter = args2
    interf_counter = args3


def build_audio(audio_type, params, filenum, audio_samples_length=-1, soi_user=""):
    '''Construct an audio signal from source files'''

    fs_output = params['fs']
    silence_length = params['silence_length']
    if audio_samples_length == -1:
        audio_samples_length = int(params['audio_length']*params['fs'])

    output_audio = np.zeros(0)
    remaining_length = audio_samples_length
    files_used = []
    clipped_files = []

    global clean_counter, noise_counter, interf_counter, users
    if audio_type == "clean":
        this_user = random.choice(users)
        while len(params['cleanfilenames_byuser'][this_user]) < 2: #we need more than 1 recording of the same user for embedding
            this_user = random.choice(users)
        source_files = params['cleanfilenames_byuser'][this_user] #make sure that these are from the same user
        idx_counter = clean_counter
    elif audio_type == "interf":
        this_user = random.choice(users)
        while this_user == soi_user: #we need it to be another user than the SOI, to avoid confusion
            this_user = random.choice(users)
        source_files = params['cleanfilenames_byuser'][this_user] #make sure that these are from the same user
        idx_counter = interf_counter
    elif audio_type == "noise":
        source_files = params['noisefilenames']
        idx_counter = noise_counter
    else:
        assert True, "invalid audio type"

    # initialize silence
    silence = np.zeros(int(fs_output*silence_length))

    # iterate through multiple clips until we have a long enough signal
    tries_left = MAXTRIES
    while remaining_length > 0 and tries_left > 0:

        # read next audio file and resample if necessary
        with idx_counter.get_lock():
            idx_counter.value += 1
            idx = idx_counter.value % np.size(source_files)

        input_audio, fs_input = audioread(source_files[idx])
        if fs_input != fs_output:
            input_audio = librosa.resample(input_audio, orig_sr=fs_input, target_sr=fs_output)

        # if current file is longer than remaining desired length, and this is
        # noise generation or this is training set, subsample it randomly
        if len(input_audio) > remaining_length and (not (audio_type == "clean" or audio_type == "interf") or not params['is_test_set']):
            idx_seg = np.random.randint(0, len(input_audio)-remaining_length)
            input_audio = input_audio[idx_seg:idx_seg+remaining_length]

        # check for clipping, and if found move onto next file
        if is_clipped(input_audio):
            clipped_files.append(source_files[idx])
            tries_left -= 1
            continue

        # concatenate current input audio to output audio stream
        files_used.append(source_files[idx])
        output_audio = np.append(output_audio, input_audio)
        remaining_length -= len(input_audio)

        # add some silence if we have not reached desired audio length
        if remaining_length > 0:
            silence_len = min(remaining_length, len(silence))
            output_audio = np.append(output_audio, silence[:silence_len])
            remaining_length -= silence_len

    if tries_left == 0:
        print("Audio generation failed for filenum " + str(filenum))
        return [], [], clipped_files
    
    if audio_type == "clean" or audio_type == "interf":
        return output_audio, files_used, clipped_files, this_user
    else:
        return output_audio, files_used, clipped_files, ""


def gen_audio(audio_type, params, filenum, audio_samples_length=-1, soi_user=""):
    '''Calls build_audio() to get an audio signal, and verify that it meets the
       activity threshold'''

    clipped_files = []
    low_activity_files = []
    if audio_samples_length == -1:
        audio_samples_length = int(params['audio_length']*params['fs'])
    
    if audio_type == "clean":
        activity_threshold = params['clean_activity_threshold']
    elif audio_type == "interf":
        activity_threshold = params['interf_activity_threshold']
    elif audio_type == "noise":
        activity_threshold = params['noise_activity_threshold']
    else:
        assert True, "invalid audio type"

    while True:
        if audio_type == "interf":
            audio, source_files, new_clipped_files, user = \
                build_audio(audio_type, params, filenum, audio_samples_length, soi_user=soi_user)
        else:
            audio, source_files, new_clipped_files, user = \
                build_audio(audio_type, params, filenum, audio_samples_length)

        clipped_files += new_clipped_files
        if len(audio) < audio_samples_length:
            continue

        if activity_threshold == 0.0:
            break

        percactive = activitydetector(audio=audio)
        if percactive > activity_threshold:
            break
        else:
            low_activity_files += source_files

    return audio, source_files, clipped_files, low_activity_files, user


def main_gen(params, filenum):
    '''Calls gen_audio() to generate the audio signals, verifies that they meet
       the requirements, and writes the files to storage'''

    print("Generating file #" + str(filenum))

    clean_clipped_files = []
    clean_low_activity_files = []
    noise_clipped_files = []
    noise_low_activity_files = []

    user = ""
    while True:
        # generate clean speech
        clean, clean_source_files, clean_cf, clean_laf, user = \
            gen_audio("clean", params, filenum)
        # generate noise
        noise, noise_source_files, noise_cf, noise_laf, _ = \
            gen_audio("noise", params, filenum, len(clean))
        
        interf_num = np.random.randint(params['interf_lower'], params['interf_upper'])
        audio_samples_length = int(params['audio_length']*params['fs'])
        interf_source_files = []
        
        clean_clipped_files += clean_cf
        clean_low_activity_files += clean_laf
        noise_clipped_files += noise_cf
        noise_low_activity_files += noise_laf

        # mix clean speech and noise
        # if specified, use specified SNR value
        if not params['randomize_snr']:
            snr = params['snr']
        # use a randomly sampled SNR value between the specified bounds
        else:
            snr = np.random.randint(params['snr_lower'], params['snr_upper'])
        
        
        sir = 100
        interfs = []
        interf_ids = []
        
        # generate interferences
        for interf_i in range(interf_num):
            interf, this_interf_source_files, interf_cf, interf_laf, interf_user = \
                gen_audio("interf", params, filenum, len(clean), soi_user=user)
            while interf_user in interf_ids:
                interf, this_interf_source_files, interf_cf, interf_laf, interf_user = \
                    gen_audio("interf", params, filenum, len(clean), soi_user=user)
            interf_ids.append(interf_user)
            interfs.append(interf)
            interf_source_files += this_interf_source_files
            
        # mix clean speech and interferences
        # if specified, use specified SIR value
        if not params['randomize_sir']:
            sir = params['sir']
        # use a randomly sampled SIR value between the specified bounds
        elif interf_num > 0:
            sir = np.random.randint(params['sir_lower'], params['sir_upper'])
            
        if not params['randomize_rt60']:
            rt60 = params['rt60']
        else:
            rt60 = (params['rt60_upper']-params['rt60_lower'])*np.random.random_sample() + params['rt60_lower']
        
        clean_snr, beam_int, beam_soi, target_level, mic = pyroom_mixer(params=params, 
                                                                  clean=clean,
                                                                  noise=noise,
                                                                  interfs=interfs,
                                                                  snr=snr,
                                                                  sir=sir,
                                                                  rt60=rt60)
        
        # unexpected clipping
        if is_clipped(clean_snr) or is_clipped(beam_int) or is_clipped(beam_soi):
            continue
        else:
            break

    # write resultant audio streams to files
    hyphen = '-'
    clean_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in clean_source_files]
    clean_files_joined = hyphen.join(clean_source_filenamesonly)[:MAXFILELEN]
    noise_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in noise_source_files]
    noise_files_joined = hyphen.join(noise_source_filenamesonly)[:MAXFILELEN]

    clean_source_files_not_used = [p for p in params['cleanfilenames_byuser'][user] if p not in clean_source_files]
    embed_source_file = random.choice(clean_source_files_not_used)

    #noisyfilename = clean_files_joined + '_' + noise_files_joined + '_snr' + \
    #                str(snr)+ '_sir' + str(sir) + '_fileid_' + str(filenum) + '.wav'
    noisyfilename = 'snr' + str(snr) + '_rt' + str(f'{rt60:.2f}') + '_ints' + str(interf_num) + '_sir' + str(sir) + '_' + clean_files_joined + '_' + noise_files_joined + '_fileid_' + str(filenum) + '.wav'
    cleanfilename = 'clean_fileid_'+str(filenum)+'.wav'
    interffilename = 'interf_fileid_'+str(filenum)+'.wav'
    micfilename = 'mic_fileid_'+str(filenum)+'.wav'

    soipath = os.path.join(params['noisyspeech_dir'], noisyfilename)
    cleanpath = os.path.join(params['clean_proc_dir'], cleanfilename)
    interfpath = os.path.join(params['noise_proc_dir'], interffilename)
    micpath = os.path.join(params['mic_proc_dir'], micfilename)

    audio_signals = [beam_soi, clean_snr, beam_int, mic]
    file_paths = [soipath, cleanpath, interfpath, micpath]
    
    for i in range(len(audio_signals)):
        try:
            audiowrite(file_paths[i], audio_signals[i], params['fs'])
        except Exception as e:
            print(str(e))
            pass

    #creating info text file with paths of clean file, beamform interference output, and interference files
    infofilename = noisyfilename.replace(".wav",".txt")
    infopath = os.path.join(params['noisyspeech_dir'], infofilename)
    info_file = open(infopath,"w")
    info_file.write(os.path.join(os.getcwd(),cleanpath)+"\n")
    info_file.write(os.path.join(os.getcwd(),embed_source_file)+"\n")
    info_file.write(os.path.join(os.getcwd(),interfpath)+"\n")
    info_file.write(os.path.join(os.getcwd(),micpath)+"\n")
    for interf_file in interf_source_files:
        info_file.write(os.path.join(os.getcwd(),interf_file)+"\n")
    info_file.close()

    return clean_source_files, clean_clipped_files, clean_low_activity_files, \
           noise_source_files, noise_clipped_files, noise_low_activity_files


def extract_list(input_list, index):
    output_list = [i[index] for i in input_list]
    flat_output_list = [item for sublist in output_list for item in sublist]
    flat_output_list = sorted(set(flat_output_list))
    return flat_output_list


def main_body():
    '''Main body of this file'''

    parser = argparse.ArgumentParser()

    # Configurations: read noisyspeech_synthesizer.cfg and gather inputs
    parser.add_argument('--cfg', default='noisyspeech_synthesizer_multimic.cfg',
                        help='Read noisyspeech_synthesizer.cfg for all the details')
    parser.add_argument('--cfg_str', type=str, default='noisy_speech')
    parser.add_argument('--set_to_create', type=str, default='None')
    args = parser.parse_args()

    params = dict()
    params['args'] = args
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f'No configuration file as [{cfgpath}]'

    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    params['cfg'] = cfg._sections[args.cfg_str]
    cfg = params['cfg']

    clean_dir = os.path.join(os.path.dirname(__file__), 'CleanSpeech')
    if cfg['speech_dir'] != 'None':
        clean_dir = cfg['speech_dir']
    if not os.path.exists(clean_dir):
        assert False, ('Clean speech data is required, '+clean_dir+' does not exists')

    noise_dir = os.path.join(os.path.dirname(__file__), 'Noise')
    if cfg['noise_dir'] != 'None':
        noise_dir = cfg['noise_dir']
    if not os.path.exists(noise_dir):
        assert False, ('Noise data is required')

    params['fs'] = int(cfg['sampling_rate'])
    params['audioformat'] = cfg['audioformat']
    params['audio_length'] = float(cfg['audio_length'])
    params['silence_length'] = float(cfg['silence_length'])
    params['total_hours'] = float(cfg['total_hours'])
    
    if cfg['fileindex_start'] != 'None' and cfg['fileindex_start'] != 'None':
        params['fileindex_start'] = int(cfg['fileindex_start'])
        params['fileindex_end'] = int(cfg['fileindex_end'])    
        params['num_files'] = int(params['fileindex_end'])-int(params['fileindex_start'])
    else:
        params['num_files'] = int((params['total_hours']*60*60)/params['audio_length'])

    params['is_test_set'] = utils.str2bool(cfg['is_test_set'])
    params['clean_activity_threshold'] = float(cfg['clean_activity_threshold'])
    params['noise_activity_threshold'] = float(cfg['noise_activity_threshold'])
    params['interf_activity_threshold'] = float(cfg['interf_activity_threshold'])
    params['snr_lower'] = int(cfg['snr_lower'])
    params['snr_upper'] = int(cfg['snr_upper'])
    params['randomize_snr'] = utils.str2bool(cfg['randomize_snr'])
    params['interf_lower'] = int(cfg['interf_lower'])
    params['interf_upper'] = int(cfg['interf_upper'])
    params['sir_lower'] = int(cfg['sir_lower'])
    params['sir_upper'] = int(cfg['sir_upper'])
    params['randomize_sir'] = utils.str2bool(cfg['randomize_sir'])
    params['room_dim_x'] = float(cfg['room_dim_x'])
    params['room_dim_y'] = float(cfg['room_dim_y'])
    params['rt60_lower'] = float(cfg['rt60_lower'])
    params['rt60_upper'] = float(cfg['rt60_upper'])
    params['randomize_rt60'] = utils.str2bool(cfg['randomize_rt60'])
    params['micdist_lower'] = float(cfg['micdist_lower'])
    params['micdist_upper'] = float(cfg['micdist_upper'])
    params['phase_threshold'] = float(cfg['phase_threshold'])
    params['target_level_lower'] = int(cfg['target_level_lower'])
    params['target_level_upper'] = int(cfg['target_level_upper'])
    
    if 'snr' in cfg.keys():
        params['snr'] = int(cfg['snr'])
    else:
        params['snr'] = int((params['snr_lower'] + params['snr_upper'])/2)

    if 'speech_csv' in cfg.keys() and cfg['speech_csv'] != 'None':
        cleanfilenames = pd.read_csv(cfg['speech_csv'])
        cleanfilenames = cleanfilenames['filename']
    else:
        cleanfilenames = glob.glob(os.path.join(clean_dir, params['audioformat']))
    params['cleanfilenames'] = cleanfilenames
    shuffle(params['cleanfilenames'])
    
    if(os.path.exists("cleanfilenames_byuser.json")):
        print("Loading data from cleanfilenames_byuser.json")
        json_file = open("cleanfilenames_byuser.json","r")
        params['cleanfilenames_byuser_full'] = json.load(json_file)
        json_file.close()
        
        users_full = list(params['cleanfilenames_byuser_full'].keys())
    else:
        print("Building cleanfilenames_byuser data")
        users_full = [os.path.basename(path).split(".wav")[0].split("_")[-2] for path in params['cleanfilenames']]
        params['cleanfilenames_byuser_full'] = {}
        user_i = 0
        user_num = len(users_full)
        for i in users_full:
            this_user_files = [s for s in params['cleanfilenames'] if "reader_"+i in s]
            params['cleanfilenames_byuser_full'][i] = this_user_files
            print(str(user_i)+"/"+str(user_num)+" : "+format((1 + user_i) / user_num, " 3.1%"), end='\r')
            user_i += 1
            #if user_i >= 10:
            #    break
        print("")
        
        print("Storing data in cleanfilenames_byuser.json")
        json_file = open("cleanfilenames_byuser.json","w")
        json.dump(params['cleanfilenames_byuser_full'], json_file, indent=4)
        json_file.close()

    global users
    
    if args.set_to_create == 'None':
        params['set_to_create'] = cfg['set_to_create']
    else:
        params['set_to_create'] = args.set_to_create
    
    if params['set_to_create'] == 'train':
        print("Creating training set...")
        params['train_perc'] = float(cfg['train_perc'])
        
        new_users_len = int(len(users_full)*params['train_perc'])
        users = users_full[:new_users_len]
        
        params['cleanfilenames_byuser'] = {k: params['cleanfilenames_byuser_full'][k] for k in users} 
        
        params['num_files'] = int(params['num_files']*params['train_perc'])
        
        params['noisyspeech_dir'] = utils.get_dir(cfg, 'noisy_destination', 'noisy')
        params['clean_proc_dir'] = utils.get_dir(cfg, 'clean_destination', 'clean')
        params['noise_proc_dir'] = utils.get_dir(cfg, 'noise_destination', 'noise')
        params['mic_proc_dir'] = utils.get_dir(cfg, 'mic_destination', 'mic')
        
    elif params['set_to_create'] == 'validtest':
        print("Creating validtest set...")
        params['train_perc'] = float(cfg['train_perc'])
        
        new_users_len = int(len(users_full)*(1-params['train_perc']))
        users = users_full[-new_users_len:]
        
        params['cleanfilenames_byuser'] = {k: params['cleanfilenames_byuser_full'][k] for k in users} 
        
        params['num_files'] = int(params['num_files']*(1-params['train_perc']))
        
        params['noisyspeech_dir'] = utils.get_dir(cfg, 'validtest_noisy_destination', 'noisy')
        params['clean_proc_dir'] = utils.get_dir(cfg, 'validtest_clean_destination', 'clean')
        params['noise_proc_dir'] = utils.get_dir(cfg, 'validtest_noise_destination', 'noise')
        params['mic_proc_dir'] = utils.get_dir(cfg, 'validtest_mic_destination', 'mic')
        
    else:
        print("Creating full set...")
        params['cleanfilenames_byuser'] = params['cleanfilenames_byuser_full']
        users = users_full
        
        params['noisyspeech_dir'] = utils.get_dir(cfg, 'noisy_destination', 'noisy')
        params['clean_proc_dir'] = utils.get_dir(cfg, 'clean_destination', 'clean')
        params['noise_proc_dir'] = utils.get_dir(cfg, 'noise_destination', 'noise')
        params['mic_proc_dir'] = utils.get_dir(cfg, 'mic_destination', 'mic')

    params['noisefilenames'] = glob.glob(os.path.join(noise_dir, params['audioformat']))
    shuffle(params['noisefilenames'])

    # Invoke multiple processes and fan out calls to main_gen() to these processes
    global clean_counter, noise_counter, interf_counter
    clean_counter = multiprocessing.Value('i', 0)
    noise_counter = multiprocessing.Value('i', 0)
    interf_counter = multiprocessing.Value('i', 0)
    
    print('Number of files to be synthesized:', params['num_files'])
    multi_pool = multiprocessing.Pool(processes=PROCESSES, initializer = init, initargs = (clean_counter, noise_counter, interf_counter, ))
    fileindices = range(params['num_files'])
    output_lists = multi_pool.starmap(main_gen, zip(repeat(params), fileindices))

    flat_output_lists = []
    num_lists = 6
    for i in range(num_lists):
        flat_output_lists.append(extract_list(output_lists, i))

    # Create log directory if needed, and write log files of clipped and low activity files
    log_dir = utils.get_dir(cfg, 'log_dir', 'Logs')

    utils.write_log_file(log_dir, 'source_files.csv', flat_output_lists[0] + flat_output_lists[3])
    utils.write_log_file(log_dir, 'clipped_files.csv', flat_output_lists[1] + flat_output_lists[4])
    utils.write_log_file(log_dir, 'low_activity_files.csv', flat_output_lists[2] + flat_output_lists[5])
    
    # Compute and print stats about percentange of clipped and low activity files
    total_clean = len(flat_output_lists[0]) + len(flat_output_lists[1]) + len(flat_output_lists[2])
    total_noise = len(flat_output_lists[3]) + len(flat_output_lists[4]) + len(flat_output_lists[5])
    pct_clean_clipped = round(len(flat_output_lists[1])/total_clean*100, 1)
    pct_noise_clipped = round(len(flat_output_lists[4])/total_noise*100, 1)
    pct_clean_low_activity = round(len(flat_output_lists[2])/total_clean*100, 1)
    pct_noise_low_activity = round(len(flat_output_lists[5])/total_noise*100, 1)
    
    print("Of the " + str(total_clean) + " clean speech files analyzed, " + str(pct_clean_clipped) + \
          "% had clipping, and " + str(pct_clean_low_activity) + "% had low activity " + \
          "(below " + str(params['clean_activity_threshold']*100) + "% active percentage)")
    print("Of the " + str(total_noise) + " noise files analyzed, " + str(pct_noise_clipped) + \
          "% had clipping, and " + str(pct_noise_low_activity) + "% had low activity " + \
          "(below " + str(params['noise_activity_threshold']*100) + "% active percentage)")


if __name__ == '__main__':
    main_body()
