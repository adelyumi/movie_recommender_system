# movie_recommender_system
Implementation of movie recommender system for PMLDL course (assignment 2)

Author: Adelina Kildeeva

Email: a.kildeeva@innopolis.university

Group: B21-DS-02

## Dataset
[MovieLens-100k](https://grouplens.org/datasets/movielens/100k/)

* It consists of 100,000 ratings from 943 users on 1682 movies
* Ratings are ranged from 1 to 5
* Each user has rated at least 20 movies
* It contains simple demographic info for the users (age, gender, occupation, zip code)

## Run the benchmark
1. Clone the repository
2. Install all dependencies
```
pip install -r requirements.txt
```
3. Run the evaluate.py
```
python benchmark/evaluate.py
```

## Benchmark scores
RMSE = 0.9834, MAE = 0.7804
