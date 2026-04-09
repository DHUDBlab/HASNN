#### HASNN
HASNN: Hierarchical Attention Spiking Neural Network for Dynamic
Graph Representation Learning(Expert Systems With Applications)

![framework](https://example.com/image.png)

#### Datasets

|             | DBLP    | Tmall     |  Patent    |
|-------------|---------|-----------| ---------- |
| #nodes      | 28,085  | 577,314   | 2,738,012  |
| #edges      | 236,894 | 4,807,545 | 13,960,811 |
| #time steps | 27      | 186       | 25         |
| #classes    | 10      | 5         | 6          |

you can download through [OneDrive](https://1drv.ms/f/c/af441250618e3e47/EogGrgPYf85NknwnETs-elYBUnIdazE-jFKgGRkGrvuEqg?e=ERxKmi)

#### Generate neighbor feature via randomwalk

```python
python generate_feature.py --dataset dblp
python generate_feature.py --dataset tmall --normalize
python generate_feature.py --dataset patent --normalize
```

#### Requirements

```python
python>=3.9
tqdm==4.67.1
scipy==1.13.1
texttable==1.7.0
torch==2.5.1+cu121 
numpy==1.22.4
numba==0.57.1
scikit_learn==1.6.1
torch_cluster (optional) for random walk sampler
```

#### Build neighborhood sampler

`python setup.py install`

#### Run

```python
# DBLP
python main.py --dataset DBLP

# Tmall
python main.py --dataset Tmall

# Patent
python main.py --dataset Patent
```

#### **parameter example**

```python
python main.py \
--dataset dblp \
--hids 128 10 \
--batch_size 1024 \
--p 0.5 \
--train_size 0.4
```

#### cite

```
@article{gan2026hasnn,
  title={HASNN: Hierarchical Attention Spiking Neural Network for Dynamic Graph Representation Learning},
  author={Gan, Yanglan and Dong, Yanzu and Guo, Wenjing and Yan, Cairong and Zou, Guobing},
  journal={Knowledge-Based Systems},
  pages={115541},
  year={2026},
  publisher={Elsevier}
}
```

