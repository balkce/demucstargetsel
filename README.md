# Target Selection Strategies for Demucs-Denoiser
Embedding- and location-based target selection strategies for the Demucs-Denoiser speech enhancement technique.

## To clone:

Before doing `git clone`, you'll need GIT LFS:

    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
    git lfs install

## Requirements

To install the requirements:

    pip install -r requirements.txt

## Creating the Training/Validation/Testing Dataset

We'll assume that the folder in which you have downloaded this repository is `/opt/demucstargetsel`.

The to-be-created dataset is based on the 2020 branch of the [Interspeech Deep Noise Suppression (DNS) Challenge](https://github.com/microsoft/DNS-Challenge). This repository also requires GIT LFS.

Once cloned, change directory to where you desire to have the base DNS code. We'll assume that is located in `/opt/DNS`:

    cd /opt/DNS
    git clone --branch interspeech2020/master https://github.com/microsoft/DNS-Challenge
    cd DNS-Challenge
    git lfs track "*.wav"
    git add .gitattributes

Then copy all the files in the repositories `datasetcreation` folder to `/opt/DNS`:

    cp /opt/demucstargetsel/datasetcreation/* .

Modify the copied files such that they make sense to your system's configuration, and run the dataset creation script:

    bash create_dns_multimic_trainvalidtest.sh

This will take a long time, depending on the amount of hours you have configured in `noisyspeech_synthesizer_multimic.cfg`.

By default, the training subset will be located in `/opt/DNS/DNS-Challenge/corpus_multimic`, while the training and validation subset in `/opt/DNS/DNS-Challenge/corpus_multimic_validtest`.

## Training

Once the dataset is created, to train will depend on what strategy you're interested in:

### Embedding-based target selection (DemucsEmbed)

Change to the `demucsembed` folder in the repository. Change the `make_dns_multiuser.sh` script to point to where the created dataset resides, and run:

    bash make_dns_multiuser.sh

Once finished, to train run:

    bash launch_embeddemucs_dns-multiuser_full64.sh

The model will be stored in `outputs/exp_dummy:embeddemucsdnsfull64/best.th`.

### Location-based target selection (DemucsPhaseBeamform)

Change to the `demucsphasebeamform` folder in the repository. Change the `make_dns_multimic.sh` script to point to where the created dataset resides, and run:

    bash make_dns_multimic.sh

Once finished, to train run:

    bash launch_demucsphase_64.sh

The model will be stored in `outputs/exp_dummy:demucsphase64/best.th`.

## Running Online

Once trained, to run online will require to first know the interface number (here assumed to be `3`) that points to PulseAudio, by running:

    python -m sounddevice

The next steps depend on what strategy you're interested in:

### Embedding-based target selection (DemucsEmbed)

First, a recording of the target speech source needs to be captured and stored in a WAV file (such as `recording.wav`) and an embedding needs to be created and stored in a JSON file (such as `embedding.json`), by running:

    python create_embedding.py recording.wav embedding.json

Then run:

    python -m denoiser.live --in 3 --out 3 --embedding embedding.json --model_path outputs/exp_dummy:embeddemucsdnsfull64/best.th --device cuda -f 4 -t 4

Alternatively, you can use the pre-trained model located in `evaluation/pretrained_models/demucsembed/best.th`

### Location-based target selection (DemucsPhaseBeamform)

Make sure that the selected device has at least two input channels. Then run:

    python -m denoiser.live --in 3 --out 3 --model_path outputs/exp_dummy:demucsphase64/best.th --device cuda -f 4 -t 4

Alternatively, you can use the pre-trained model located in `evaluation/pretrained_models/denoiserphase/best.th`


## Evaluation

Change to the repositories `evaluation` directory:

    cd evaluation

Modify the `create_dataset_wavs_DNSMultiMic.py` to point to where the created dataset resides and run:

    python create_dataset_wavs_DNSMultiMic.py

Copy the trained models to their respective folder inside `evaluation/pretrained_models` and run:

    python SEEval_DNSMultiMic_script.py


