import logging
import os
import json

import hydra

logger = logging.getLogger(__name__)

def build_userinfo(user_info_json,full_json):
  assert os.path.exists(full_json), full_json+" does not exists."
  
  logger.info("\t Reading full json file...")
  full_json_file = open(full_json,"r")
  full_jsons = json.load(full_json_file)
  full_json_file.close()
  logger.info("\t\t done.")
  
  full_jsons_len = len(full_jsons)
  
  logger.info("\t Full json length:           "+str(full_jsons_len))
  
  logger.info("\t Extracting users from full json...")
  wav_files = [j[2] for j in full_jsons]
  users = []
  for wav_path in wav_files:
    if "silence" in wav_path:
      user_id = "silence"
    else:
      user_id = wav_path.split("reader_")[1].split("_")[0]
    if user_id not in users:
      users.append(user_id)
  logger.info("\t Users in full json:       "+str(len(users)))
  
  user_num = len(users)
  
  user_dic = {}
  
  logger.info("\t Creating user info from full json...")
  mini_jsons = []
  for user in users:
    if user == "silence":
      user_jsons = [json for json in full_jsons if "silence" in json[2]]
    else:
      user_jsons = [json for json in full_jsons if "reader_"+user in json[2]]
    user_dic[user] = user_jsons
  logger.info("\t\t done.")
  
  logger.info("\t Storing in mini json file.")
  userinfo_json_file = open(user_info_json,"w")
  json.dump(user_dic, userinfo_json_file, indent=4)
  userinfo_json_file.close()

def build_userinfo_jsons(args):
  this_train_json = args.dset.dset.train
  full_train_json = this_train_json.split(".json")[0]+"_full.json"
  userinfo_train_json = this_train_json.split(".json")[0]+"_userinfo.json"
  logger.info("Training full json:   "+full_train_json)
  logger.info("Training user info:   "+userinfo_train_json)
  build_userinfo(userinfo_train_json, full_train_json)
  
  this_valid_json = args.dset.dset.valid
  full_valid_json = this_valid_json.split(".json")[0]+"_full.json"
  userinfo_valid_json = this_valid_json.split(".json")[0]+"_userinfo.json"
  logger.info("Validation full json:   "+full_valid_json)
  logger.info("Validation user info:   "+userinfo_valid_json)
  build_userinfo(userinfo_valid_json, full_valid_json)
  
  this_test_json = args.dset.dset.test
  full_test_json = this_test_json.split(".json")[0]+"_full.json"
  userinfo_test_json = this_test_json.split(".json")[0]+"_userinfo.json"
  logger.info("Testing full json:   "+full_test_json)
  logger.info("Testing user info:   "+userinfo_test_json)
  build_userinfo(userinfo_test_json, full_test_json)

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
    
    build_userinfo_jsons(args)


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
