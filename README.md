# Fractional Fourier Transform Meets Transformer Encoder

In this study, we introduce fractional Fourier transform version of the [FNet](https://github.com/google-research/google-research/tree/master/f_net) model, which is called FrFNet. Instead of the attention layer in transformer encoders, we use fractional Fourier transform layer. We can obtain better GLUE benchmark scores by optimizing fraction values. Our paper is published in IEEE Signal Processing Letters. Early access version of the paper: https://ieeexplore.ieee.org/document/9931916.


## Pre-training and Fine-tuning

This repository is based on [FNet](https://github.com/google-research/google-research/tree/master/f_net) repository and contains necessary modifications for implementation and adaptation of fractional Fourier transform to transformer encoder structure. For installation of the required libraries please refer to [original repository](https://github.com/google-research/google-research/tree/master/f_net).

For both training scheme [SentencePiece vocabulary model](https://storage.googleapis.com/gresearch/f_net/vocab/c4_bpe_sentencepiece.model) should be downloaded. Assuming the current directory points out this repository, you can start training with the following command line:

```
python main.py --workdir=$WORKING_DIRECTORY --vocab_filepath=$VOCAB_FILE_PATH --config=$CONFIGURATION_FILE_PATH
```

Here, ```$WORKING_DIRECTORY``` determines the directory where checkpoints and other necessary files will be saved. ```$VOCAB_FILE_PATH``` indicates the directory of the SentencePiece vocabulary model and ```$CONFIGURATION_FILE_PATH``` depends on training scheme and directs to either pretraining.py or classification.py in the configs folder.

For example (assuming the current directory is this repository),

```
python main.py --workdir=./pretrainings/frac_order_0.75/ --vocab_filepath=./vocab/c4_bpe_sentencepiece.model --config=./configs/pretrainig.py
```

## Adjusting the fractional order and FrFNet model configuration

For training by FrFNet configuration, in configs/base.py, please adjust model architecture as ```config.model_arch: ModelArchitecture = ModelArchitecture.FRAC_NET``` (default configuration) and set config.frac_order as desired fraction value. Other model parameters can also be customized from the same file.

## Bibtex

```
@article{sahinuc2022fractional,  
  author={\c{S}ahinu\c{c}, Furkan and Ko\c{c}, Aykut},  
  journal={IEEE Signal Processing Letters},   
  title={Fractional Fourier Transform Meets Transformer Encoder},   
  year={2022},
  doi={10.1109/LSP.2022.3217975}
}
```
