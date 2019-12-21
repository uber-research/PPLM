

Begin setup by running the executable 'setup_script.sh'. The above script installs requirements, downloads the gpt2-model,and then moves it to the relevant location. With python 3.7, follow instructions below for pplm control:

## Example command for bag-of-words control ####### 

python pplm.py -B words/space.txt --cond-text "The president" --length 100 --gamma 1.5 --num-iterations 3 --num-samples 10 --stepsize 0.01 --window-size 5 --fusion-kl-scale 0.01 --fusion-gm-scale 0.01

## Tuning hyperparameters for bag-of-words control
1. Reduce stepsize to decrease conditioning, and vice verse.

2. If the language being generated is repetitive (For e.g. science science experiment experiment), there are several options to consider: </br>
	a) Reduce the stepsize </br>
	b) Increase the kl-loss coefficient or decrease the gm-scaling term </br>
	c) Add "--grad-length xx" where xx is an (integer <= length, e.g. --grad-length 30).</br>

## Example command for discriminator based sentiment control
python pplm.py -D sentiment --label-class 3 --cond-text "The lake" --length 10 --gamma 1.0 --num-iterations 30 --num-samples 10 --stepsize 0.01 --fusion-kl-scale 0.01 --fusion-gm-scale 0.95

## Tuning hyperparameters for discriminator based sentiment control
1. Reduce stepsize to decrease conditioning, and vice verse.

2. Use label-class 3 for negative, and label-class 2 for positive

## Example command for detoxificiation:
python pplm.py -D toxicity --length 100 --num-iterations 10 --cond-text 'TH PEOPLEMan goddreams Blacks' --gamma 1.0 --num-samples 10 --stepsize 0.02



The code in the folder 'pytorch-pretraiend-bert' is from the repository: https://github.com/huggingface/transformers, with edits to propagate the probability vector through the embedding matrix.
