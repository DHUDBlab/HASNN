#### Datasets

| DBLP        | Tmall   | Patent    |            |
| #nodes      | 28,085  | 577,314   | 2,738,012  |
| #edges      | 236,894 | 4,807,545 | 13,960,811 |
| #time steps | 27      | 186       | 25         |
| #classes    | 10      | 5         | 6          |

you can download through link()

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
```

#### Build neighborhood sampler

`python setup.py install`

#### Run HASNN

```python
# DBLP
python main.py --dataset dblp --hids 512 64 --batch_size 1024 --p 0.3 --train_size 0.4 --dropout 0.7
python main.py --dataset dblp --hids 512 64 --batch_size 1024 --p 0.3 --train_size 0.6 --dropout 0.7
python main.py --dataset dblp --hids 512 64 --batch_size 1024 --p 0.3 --train_size 0.8 --dropout 0.7

# Tmall
python main.py --dataset tmall --hids 512 64 --batch_size 1024 --p 1.0 --train_size 0.4 --dropout 0.8
python main.py --dataset tmall --hids 512 64 --batch_size 1024 --p 1.0 --train_size 0.6 --dropout 0.8
python main.py --dataset tmall --hids 512 64 --batch_size 1024 --p 1.0 --train_size 0.8 --dropout 0.8

# Patent
python main.py --dataset patent --hids 512 64 --batch_size 2048 --p 1.0 --train_size 0.8 --dropout 0.6
python main.py --dataset patent --hids 512 64 --batch_size 2048 --p 1.0 --train_size 0.8 --dropout 0.6
python main.py --dataset patent --hids 512 64 --batch_size 2048 --p 1.0 --train_size 0.8 --dropout 0.6

```

