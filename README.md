# TIGER
[![arXiv](https://img.shields.io/badge/arXiv-2305.05065-red.svg)](https://arxiv.org/abs/2305.05065)

This is the pytorch implementation of the paper at NeurIPS 2023:
> [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
> 
> Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy.

## Usage

### Data
The experimental datasets should be preprocessed into **JSON format**. You may refer to this [example data](https://github.com/HonghuiBao2000/LETTER/tree/master/data) for guidance.
### Training & Evaluation
#### 1. Train the RQ-VAE Model
```
python run_gr_id.py
```
#### 2. Train the T5 Model with Online Tokenization
Once the RQ-VAE model is trained, you can proceed to train the T5 model using online tokenization (i.e., tokenization is performed during training, rather than stored offline):
```
python run_gr_rec.py
```

## Note

This project is based on the [LETTER](https://github.com/HonghuiBao2000/LETTER) repository, and is compatible with using LETTER as a tokenizer. However, unlike LETTER which removes duplicates through **post-processing**, our implementation introduces **deduplication directly via suffix tokens during token generation**.
