import csv
import io
import json
import logging
import random
from datetime import datetime, timedelta

import boto3
import numpy as np
import pandas as pd

from airflow.sdk import DAG
from airflow.operators.python import PythonOperator

S3_BUCKET = "phirisolomonvst"
RATINGS_KEY = "ml-1m/ratings.dat"
MOVIES_KEY = "ml-1m/movies.dat"
EMBEDDINGS_PREFIX = "ml-1m/embeddings"
USER_EMB_PREFIX = "ml-1m/user_embeddings"
RECS_PREFIX = "ml-1m/recommendations"

PARTITION_BOUNDARIES = [
    ("part1", 956620800,  965001599),
    ("part2", 965001600,  972691199),
    ("part3", 972691200,  975110399),
    ("part4", 975110400, 9999999999),
]

TOP_N = 5
TOP_PERCENTILE = 5


def _s3():
    return boto3.client("s3")


def _load_ratings(end_ts):
    body = _s3().get_object(Bucket=S3_BUCKET, Key=RATINGS_KEY)["Body"].read().decode("latin-1")
    rows = []
    for line in body.strip().split("\n"):
        parts = line.strip().split("::")
        if len(parts) == 4 and int(parts[3]) <= end_ts:
            rows.append({
                "user_id":   int(parts[0]),
                "movie_id":  int(parts[1]),
                "rating":    float(parts[2]),
                "timestamp": int(parts[3]),
            })
    return pd.DataFrame(rows)


def _load_movies():
    body = _s3().get_object(Bucket=S3_BUCKET, Key=MOVIES_KEY)["Body"].read().decode("latin-1")
    rows = []
    for line in body.strip().split("\n"):
        parts = line.strip().split("::")
        if len(parts) >= 2:
            rows.append({"movie_id": int(parts[0]), "title": parts[1]})
    return pd.DataFrame(rows)


def _load_movie_embeddings():
    s3 = _s3()
    id_text = s3.get_object(Bucket=S3_BUCKET, Key=f"{EMBEDDINGS_PREFIX}/movie_id_list.txt")["Body"].read().decode("utf-8")
    movie_ids = [int(l.strip()) for l in id_text.strip().split("\n") if l.strip()]
    vec_bytes = s3.get_object(Bucket=S3_BUCKET, Key=f"{EMBEDDINGS_PREFIX}/movie_vecs.npy")["Body"].read()
    matrix = np.load(io.BytesIO(vec_bytes), allow_pickle=False)
    return {movie_ids[i]: matrix[i] for i in range(len(movie_ids))}


def _load_user_embeddings(label):
    s3 = _s3()
    user_ids = np.load(io.BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=f"{USER_EMB_PREFIX}/{label}/user_ids.npy")["Body"].read()), allow_pickle=False)
    emb_mat = np.load(io.BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=f"{USER_EMB_PREFIX}/{label}/embeddings.npy")["Body"].read()), allow_pickle=False)
    return user_ids, emb_mat


def _cosine_sim(vec, matrix):
    vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
    mat_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return mat_norm @ vec_norm


def _recommend_by_similarity(user_vec, movie_embeddings, already_rated):
    candidates = [mid for mid in movie_embeddings if mid not in already_rated]
    if not candidates:
        return []
    mat = np.stack([movie_embeddings[mid] for mid in candidates])
    sims = _cosine_sim(user_vec, mat)
    return [candidates[i] for i in np.argsort(sims)[::-1][:TOP_N]]


def _recommend_by_popularity(ratings, already_rated):
    counts = (
        ratings[~ratings["movie_id"].isin(already_rated)]
        .groupby("movie_id")["rating"].count()
        .sort_values(ascending=False)
    )
    return list(counts.index[:TOP_N])


def generate_recommendations(partition_index, **context):
    label, _, end_ts = PARTITION_BOUNDARIES[partition_index]
    timestamp_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    ratings = _load_ratings(end_ts)
    title_map = dict(zip(*[_load_movies()[c].values for c in ["movie_id", "title"]]))
    movie_embs = _load_movie_embeddings()

    try:
        user_ids_arr, user_emb_mat = _load_user_embeddings(label)
        user_emb_map = {int(uid): user_emb_mat[i] for i, uid in enumerate(user_ids_arr)}
    except Exception as e:
        logging.warning("Could not load user embeddings: %s", e)
        user_emb_map = {}

    rating_counts = ratings.groupby("user_id")["rating"].count()

    cold_recs = _recommend_by_popularity(ratings, set())
    cold_titles = [title_map.get(mid, "Unknown") for mid in cold_recs]

    threshold = rating_counts.quantile(1.0 - TOP_PERCENTILE / 100.0)
    top_users = rating_counts[rating_counts >= threshold].index.tolist()
    random.seed(partition_index)
    top_uid = random.choice(top_users) if top_users else None

    if top_uid is not None:
        already_rated = set(ratings[ratings["user_id"] == top_uid]["movie_id"].values)
        last_ts = int(ratings[ratings["user_id"] == top_uid]["timestamp"].max())
        last_interaction = datetime.utcfromtimestamp(last_ts).isoformat()
        num_ratings = int(rating_counts.get(top_uid, 0))
        top_recs = _recommend_by_similarity(user_emb_map[top_uid], movie_embs, already_rated) if top_uid in user_emb_map else _recommend_by_popularity(ratings, already_rated)
    else:
        top_uid, top_recs, last_interaction, num_ratings = "N/A", [], "N/A", 0

    top_titles = [title_map.get(mid, "Unknown") for mid in top_recs]

    records = [
        {
            "User_Type": "cold", "User_ID": -1, "Last_Interaction_Time": "N/A",
            "Num_Ratings_Observed": 0, "Partition": label,
            "Recommended_Movie_IDs": json.dumps(cold_recs),
            "Recommended_Movie_Titles": json.dumps(cold_titles),
        },
        {
            "User_Type": "top", "User_ID": top_uid, "Last_Interaction_Time": last_interaction,
            "Num_Ratings_Observed": num_ratings, "Partition": label,
            "Recommended_Movie_IDs": json.dumps(top_recs),
            "Recommended_Movie_Titles": json.dumps(top_titles),
        },
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

    _s3().put_object(
        Bucket=S3_BUCKET,
        Key=f"{RECS_PREFIX}/{label}/recommendations_{timestamp_str}.csv",
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    logging.info("Saved recommendations for %s", label)


with DAG(
    dag_id="step4_recommendations",
    start_date=datetime(2026, 6, 6),
    schedule="0 */12 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["step4", "hw4"],
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:

    t1 = PythonOperator(task_id="part1", python_callable=generate_recommendations, op_kwargs={"partition_index": 0})
    t2 = PythonOperator(task_id="part2", python_callable=generate_recommendations, op_kwargs={"partition_index": 1})
    t3 = PythonOperator(task_id="part3", python_callable=generate_recommendations, op_kwargs={"partition_index": 2})
    t4 = PythonOperator(task_id="part4", python_callable=generate_recommendations, op_kwargs={"partition_index": 3})

    t1 >> t2 >> t3 >> t4
