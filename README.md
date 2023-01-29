# [WWW'23] KHAN: Knowledge-Aware Hierarchical Attention Networks for Accurate Political Stance Prediction
This repository provides an implementation of *KHAN* as described in the paper: [KHAN: Knowledge-Aware Hierarchical Attention Networks for Accurate Political Stance Prediction](https://yy-ko.github.io/assets/files/WWW23-khan-paper.pdf) by Yunyong Ko, Seongeun Ryu, Soeun Han, Youngseung Jeon, Jaehoon Kim, Sohyun Park, Kyungsik Han, Hanghang Tong, and Sang-Wook Kim, In Proceedings of the ACM Web Conference (WWW) 2023.

## The overview of KHAN
![The overview of KHAN](./assets/khan_overview.png)

## Available dataset
1. [Reddit Dataset | Conservative and Liberal](https://drive.google.com/drive/folders/1RDSp2SoGgRFGybarVFUo6OPth8fiGjUC)
2. [FB15k Dataset | Papers With Code](https://paperswithcode.com/dataset/fb15k)
3. [DBpedia Dataset | Papers With Code](https://paperswithcode.com/dataset/dbpedia)
4. [YAGO Dataset | Papers With Code](https://paperswithcode.com/dataset/yago)
5. [NELL Dataset | Papers With Code](https://paperswithcode.com/dataset/nell)
6. [Wikidata:Every politician/Political data model](https://www.wikidata.org/wiki/Wikidata:WikiProject_every_politician)

## Datasets
1. [News articles datasets](https://drive.google.com/drive/u/2/folders/1ksV0PUncXyBnEHGPB4H4mae2ybXX3Ch0) (SemEval, AllSides-S, AllSides-L)

|Dataset|# of articles|Class distribution|
|:---:|:---:|:---:|
|SemEval|645|407/238|
|AllSides-S|14.7k|6.6k/4.6k/3.5k|
|AllSides-L|719.2k|112.4k/202.9k/99.6k/62.6k/241.5k|

2. [Political Knowledge Graphs](https://drive.google.com/drive/u/2/folders/1DHlKOhKgISw9VTYmbMvnsIbaaLRtqhbq) (KG-conservative, KG-liberal)
3. [Pre-trained KG embeddings](https://drive.google.com/drive/u/2/folders/14EgeI1RdSTccETqRgDd36writP6lUu1R) (common, conservative, liberal)

## File structure
```
├── KHAN
      ├── datasets             # data for KHAN, you can download from above google drive link
            ├── article data
                  ├── SemEval
                  ├── Allsides-S
                  └── Allsides-L
                        ├── train
                        └── test
            ├── KG data
                  ├── KG-conservative
                        ├── entities_con.dict
                        ├── relations_con.dict
                        └── triplets_con.txt
                  └── KG-liberal
                        ├── entities_lib.dict
                        ├── relations_lib.dict
                        └── triplets_lib.txt
      └── pre-trained
            ├── common_emb
            ├── conservative_emb
            └── liberal_emb

```

## Dependencies
Our code runs on the Intel i7-9700k CPU with 64GB memory and NVIDIA RTX 2080 Ti GPU with 12GB, with the following packages installed:
```
python 3.8.10
torch 1.11.0
torchtext 0.12.0
pandas
numpy
argparse
sklearn
```

## How to run
```
python3 main.py \
  --gpu_index=0 \
  --batch_size=16 \
  --num_epochs=50 \
  --learning_rate=0.001 \
  --max_sentence=20 \
  --embed_size=256 \
  --dropout=0.3 \
  --num_layer=1 \
  --num_head=4 \
  --d_hid=128 \
  --dataset=SEMEVAL \
  --alpha=0.6 \
  --beta=0.2
```


## Citation
Please cite our paper if you have used the code in your work. You can use the following BibTex citation:
```
@inproceedings{ko2023khan,
  title={KHAN: Knowledge-Aware Hierarchical Attention Networks for Accurate Political Stance Prediction},
  author={Ko, Yunyong and Ryu, Seongeun and Han, Soeun and Jeon,Youngseung and Kim, Jaehoon and Park, Sohyun and Han, Kyungsik Tong, Hanghang and Kim., Sang-Wook},
  booktitle={Proceedings of the ACM Web Conference (WWW) 2023},
  pages={xxxx--xxxx},
  year={2023}
}
```
