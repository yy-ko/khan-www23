# KHAN: Knowledge-Aware Hierarchical Attention Networks

## Available dataset
1. [Reddit Dataset | Conservative and Liberal](https://drive.google.com/drive/folders/1RDSp2SoGgRFGybarVFUo6OPth8fiGjUC)
2. [FB15k Dataset | Papers With Code](https://paperswithcode.com/dataset/fb15k)
3. [DBpedia Dataset | Papers With Code](https://paperswithcode.com/dataset/dbpedia)
4. [YAGO Dataset | Papers With Code](https://paperswithcode.com/dataset/yago)
5. [NELL Dataset | Papers With Code](https://paperswithcode.com/dataset/nell)
6. [Wikidata:Every politician/Political data model](https://www.wikidata.org/wiki/Wikidata:WikiProject_every_politician)

## Requirements

## Usage
```
python3 main.py \
  --gpu_index=1 \
  --batch_size=16 \
  --eval_batch_size=16 \ 
  --embed_size=128 \
  --num_epochs=100 \
  --learning_rate=0.5 \
  --dataset=ALLSIDES \
  --max_len=256
```
