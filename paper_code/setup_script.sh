#!/bin/bash

python -m pip install requirements.txt
git clone https://github.com/circulosmeos/gdown.pl.git
cd gdown.pl
./gdown.pl https://drive.google.com/open?id=15TvAxA8TS8nn1lCzpVPn-Myp5RDlJiHF gpt2
unzip gpt2
mv gpt2_pt_models ../.
cd ..
