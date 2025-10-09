<div align="center">

# Deep Cross-modal Evidential Learning for Composed Noisy Video Retrieval

</div>

## Description
This repository contains the code for the paper ["Deep Cross-modal Evidential Learning for Composed Noisy Video Retrieval"]

This repository contains: 

```markdown
📦 CURE
 ┣ 📂 configs                 # hydra config files
 ┣ 📂 src                     # Pytorch datamodules
 ┣ 📂 tools                   # scrips and notebooks
 ┣ 📜 .gitignore
 ┣ 📜 README.md
 ┣ 📜 test.py
 ┗ 📜 train.py

 ```

## Installation :construction_worker:

<details><summary>Create environment</summary>
&emsp; 

```bash
conda create --name cure
conda activate cure
```

To install the necessary packages, you can use the provided requirements.txt file:
```bash
python -m pip install -r requirements.txt
```

or 
```bash
conda env create -f environment.yml
```

The code was tested on Python 3.10 and PyTorch 2.4.

</details>

<details><summary>Download the datasets</summary>

### WebVid-CoVR
To use the WebVid-CoVR dataset, you will have to download the WebVid videos and merge annotation.

to merge the annotation
```bash
cat annotation/webvid-covr2m/all_csvs.zip.* > annotation/webvid-covr2m/all_csvs.zip
unzip annotation/webvid-covr2m/all_csvs.zip
```

To download the videos, install [`mpi4py`](https://mpi4py.readthedocs.io/en/latest/install.html#) (``conda install -c conda-forge mpi4py``) and run:
```bash
ln -s /path/to/your/datasets/folder datasets
python tools/scripts/download_covr.py
```

</details>



## Usage :computer:
<details><summary>Computing BLIP embeddings</summary>
&emsp; 

Before training, you will need to compute the BLIP embeddings for the videos/images. To do so, run:
```bash
# This will compute the BLIP embeddings for the WebVid-CoVR videos. 
python tools/embs/save_blip_embs_vids.py --video_dir datasets/WebVid/2M --todo_ids annotation/webvid-covr2m/webvid2m-covr_train.csv 

# This will compute the BLIP embeddings for the Test videos.
python tools/embs/save_blip_embs_vids.py --video_dir datasets/WebVid/2M --todo_ids annotation/webvid-covr2m/webvid2m-covr_test.csv 

# This will compute the Multimodal BLIP embeddings for the WebVid-CoVR videos. 
python tools/embs/save_blip_embs_vids_multimodal.py --video_dir datasets/WebVid/2M --todo_ids annotation/webvid-covr2m/webvid2m-covr_train.csv 

```

&emsp; 
</details>


<details><summary>Training</summary>
&emsp; 

The command to launch a training experiment is the folowing:
```bash
python train.py [OPTIONS]
```
The parsing is done by using the powerful [Hydra](https://github.com/facebookresearch/hydra) library. You can override anything in the configuration by passing arguments like ``foo=value`` or ``foo.bar=value``. See *Options parameters* section at the end of this README for more details.

&emsp; 
</details>

<details><summary>Evaluating</summary>
&emsp; 

The command to evaluate is the folowing:
```bash
python test.py test=<test> [OPTIONS]
```
&emsp; 
</details>


## Acknowledgements
Based on [BLIP](https://github.com/salesforce/BLIP/) and [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/tree/main).

