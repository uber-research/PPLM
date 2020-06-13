# PPLM

This repository contains code to run the Plug and Play Language Model (PPLM), as described in this **[blog post](https://eng.uber.com/pplm)** and **[arXiv paper](https://arxiv.org/abs/1912.02164)**. A **[demo](https://transformer.huggingface.co/model/pplm)** and **[Colab notebook](https://colab.research.google.com/drive/1Ux0Z4-ruiVtJ6jUk98uk6FqfvGHCOYL3)** are also available.


Note: If you are planning on using PPLM as a baseline, and would like to use the parameters listed in the paper's Appendix, please use the LM and the discriminator from this **[folder](https://github.com/uber-research/PPLM/tree/master/paper_code)**.
Alternatively, tune the hyperparamters on your own if you are using the code/models in the main directory and/or the **[🤗/Transformers](https://transformer.huggingface.co/model/pplm)** for a fair comparison (the optimal parameters for these models/discriminators are roughly off by a factor of 5 from those used in the paper).

PPLM is also integrated into the **[🤗/Transformers](https://github.com/huggingface/transformers/tree/master/examples/pplm)** repository.

![header image](./imgs/headfigure.png)

## Plug and Play Language Models: a Simple Approach to Controlled Text Generation
Authors: [Sumanth Dathathri](https://dathath.github.io/), [Andrea Madotto](https://andreamad8.github.io/), Janice Lan, Jane Hung, Eric Frank, [Piero Molino](https://w4nderlu.st/), [Jason Yosinski](http://yosinski.com/), and [Rosanne Liu](http://www.rosanneliu.com/)

PPLM allows a user to flexibly plug in one or more tiny attribute models representing the desired steering objective into a large, unconditional language model (LM). The method has the key property that it uses the LM _as is_—no training or fine-tuning is required—which enables researchers to leverage best-in-class LMs even if they do not have the extensive hardware required to train them.

See also our [arXiv paper](https://arxiv.org/abs/1912.02164), [blog post](https://eng.uber.com/pplm), and try it out for yourself with no setup using the [Colab notebook](https://colab.research.google.com/drive/1Ux0Z4-ruiVtJ6jUk98uk6FqfvGHCOYL3).

## Setup

```bash
pip install -r requirements.txt
```

## Citation
```
@inproceedings{
Dathathri2020Plug,
title={Plug and Play Language Models: A Simple Approach to Controlled Text Generation},
author={Sumanth Dathathri and Andrea Madotto and Janice Lan and Jane Hung and Eric Frank and Piero Molino and Jason Yosinski and Rosanne Liu},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=H1edEyBKDS}
}
```

## PPLM-BoW 

### Example command for bag-of-words control

```bash
python run_pplm.py -B military --cond_text "The potato" --length 50 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.03 --window_length 5 --kl_scale 0.01 --gm_scale 0.99 --colorama --sample
```

### Tuning hyperparameters for bag-of-words control

1. Increase `--stepsize` to intensify topic control, and decrease its value to soften the control. `--stepsize 0` recovers the original uncontrolled GPT-2 model. 

2. If the language being generated is repetitive (For e.g. "science science experiment experiment"), there are several options to consider: </br>
	a) Reduce the `--stepsize` </br>
	b) Increase `--kl_scale` (the KL-loss coefficient) or decrease `--gm_scale` (the gm-scaling term) </br>
	c) Add `--grad-length xx` where xx is an (integer <= length, e.g. `--grad-length 30`).</br>


## PPLM-Discrim

### Example command for discriminator based sentiment control

```bash
python run_pplm.py -D sentiment --class_label 2 --cond_text "My dog died" --length 50 --gamma 1.0 --num_iterations 10 --num_samples 10 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample
```

### Tuning hyperparameters for discriminator control

1. Increase `--stepsize` to intensify topic control, and decrease its value to soften the control. `--stepsize 0` recovers the original uncontrolled GPT-2 model. 

2. Use `--class_label 3` for negative, and `--class_label 2` for positive


The discriminator and the GPT-2 model in the root directory are different from those used for the analysis in the paper. Code and models corresponding to the paper can be found [here](https://github.com/uber-research/PPLM/tree/master/paper_code).
