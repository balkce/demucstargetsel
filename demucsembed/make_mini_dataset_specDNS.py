import logging
import os
import json
import random
import glob

import hydra

logger = logging.getLogger(__name__)

def get_users(path,user):
  path_users = []
  path_users.append(user)
  
  #extract interferences from txt file
  rec_dir = os.path.dirname(path)
  info_txt_path = path.replace(".wav",".txt")
  info_txt = open(info_txt_path, "r")
  info_lines = info_txt.readlines()
  info_txt.close()
  
  for i in range(2,len(info_lines)):
    interf_path = info_lines[i]
    rec_name = os.path.basename(interf_path)
    #extract user id from filename similar to:
    # book_11308_chp_0034_reader_00335_42.wav
    user_id = rec_name.split("reader_")[1].split("_")[0]
    path_users.append(user_id)
  return path_users

def build_mini_json(mini_json,full_json,user_json,k=1):
  assert os.path.exists(full_json), full_json+" does not exists."
  
  logger.info("\t Reading full json file...")
  full_json_file = open(full_json,"r")
  full_jsons = json.load(full_json_file)
  full_json_file.close()
  logger.info("\t\t done.")
  
  full_jsons_len = len(full_jsons)
  
  logger.info("\t Full json length:           "+str(full_jsons_len))
  
  logger.info("\t Reading user json file...")
  user_json_file = open(user_json,"r")
  user_dic = json.load(user_json_file)
  user_json_file.close()
  logger.info("\t\t done.")
  
  users = list(user_dic.keys())
  
  user_num = len(users)
  
  logger.info("\t Randomly selecting 1 json per user to create mini json...")
  mini_jsons = []
  for user in users:
    user_jsons = user_dic[user]
    thisuser_jsons = random.choices(user_jsons,k=k)
    mini_jsons += thisuser_jsons
  logger.info("\t\t done.")
  
  logger.info("\t Mini json length:       "+str(len(mini_jsons)))
  
  random.shuffle(mini_jsons)
  
  logger.info("\t Storing in mini json file.")
  mini_json_file = open(mini_json,"w")
  json.dump(mini_jsons, mini_json_file, indent=4)
  mini_json_file.close()

def build_mini_jsons(args):
  this_train_json = args.dset.dset.train
  full_train_json = this_train_json.split(".json")[0]+"_full.json"
  userinfo_train_json = this_train_json.split(".json")[0]+"_userinfo.json"
  logger.info("Training full json:   "+full_train_json)
  logger.info("Training mini json:   "+this_train_json)
  build_mini_json(this_train_json, full_train_json, userinfo_train_json,k=5)
  
  this_valid_json = args.dset.dset.valid
  full_valid_json = this_valid_json.split(".json")[0]+"_full.json"
  userinfo_valid_json = this_valid_json.split(".json")[0]+"_userinfo.json"
  logger.info("Validation full json:   "+full_valid_json)
  logger.info("Validation mini json:   "+this_valid_json)
  build_mini_json(this_valid_json, full_valid_json, userinfo_valid_json,k=1)
  
  this_test_json = args.dset.dset.test
  if (os.path.exists(this_test_json)):
    logger.info("Testing mini json:   "+this_test_json)
    logger.info("\t already exists.")
    logger.info("Testing data should be kept the same throughout training.")
    logger.info("\t not modifying it.")
  else:
    full_test_json = this_test_json.split(".json")[0]+"_full.json"
    userinfo_test_json = this_test_json.split(".json")[0]+"_userinfo.json"
    logger.info("Testing full json:   "+full_test_json)
    logger.info("Testing mini json:   "+this_test_json)
    build_mini_json(this_test_json, full_test_json, userinfo_test_json,k=1)

def _main(args):
    global __file__
    # Updating paths in config
    for key, value in args.dset.dset.items():
        if isinstance(value, str):
            args.dset.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)
    
    logger.debug(args)
    
    build_mini_jsons(args)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)

if __name__ == "__main__":
    main()
