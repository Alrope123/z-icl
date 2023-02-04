#!/bin/sh
pip install simcse==0.4
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install gdown==4.6.0
pip install datasets==2.3.0
pip install nltk==3.8.1
pip install tqdm==4.64.1
pip install transformers==4.26.0
conda install -c pytorch faiss-cpu==1.7.1