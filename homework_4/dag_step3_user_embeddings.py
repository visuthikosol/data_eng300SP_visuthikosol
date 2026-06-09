import io
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
EMBEDDINGS_PREFIX = "ml-1m/embeddings"
USER_EMBEDDINGS_PREFIX = "ml-1m/user_embeddings"

PARTITION_BOUNDARIES = [
    ("part1", 956620800,  965001599),
    ("part2", 965001600,  972691199),
    ("part3", 972691200,  975110399),
    ("part4", 975110400, 9999999999),
]

SAMPLE_FRACTION = 0.30


def _load_ratings():
    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=S3_BUCKET, Key=RATINGS_KEY)["Body"].read().decode("latin-1")
    rows = []
    for line in body.strip().split("\n"):
        parts = line.strip().split("::")
        if len(parts) == 4:
            rows.append({
                "user_id":   int(parts[0]),
                "movie_id":  int(parts[1]),
                "rating":    float(parts[2]),
                "timestamp": int(parts[3]),
            })
    return pd.DataFrame(rows)


def _load_movie_embeddings():
    s3 = boto3.client("s3")
    id_text = s3.get_object(Bucket=S3_BUCKET, Key=f"{EMBEDDINGS_PREFIX}/movie_id_list.txt")["Body"].read().decode("utf-8")
    movie_ids = [int(l.strip()) for l in id_text.strip().split("\n") if l.strip()]
    vec_bytes = s3.get_object(Bucket=S3_BUCKET, Key=f"{EMBEDDINGS_PREFIX}/movie_vecs.npy")["Body"].read()
    matrix = np.load(io.BytesIO(vec_bytes), allow_pickle=False)
    return {movie_ids[i]: matrix[i] for i in range(len(movie_ids))}


def _save_npy(arr, key):
    s3 = boto3.client("s3")
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


def compute_embeddings(partition_index, **context):
    label, start_ts, end_ts = PARTITION_BOUNDARIES[partition_index]

    all_ratings = _load_ratings()
    cumulative = all_ratings[all_ratings["timestamp"] <= end_ts].copy()
    current = all_ratings[(all_ratings["timestamp"] >= start_ts) & (all_ratings["timestamp"] <= end_ts)].copy()

    if partition_index == 0:
        previous_users = set()
    else:
        prev_end = PARTITION_BOUNDARIES[partition_index - 1][2]
        previous_users = set(all_ratings[all_ratings["timestamp"] <= prev_end]["user_id"].unique())

    new_users = set(current["user_id"].unique()) - previous_users

    all_users = list(cumulative["user_id"].unique())
    random.seed(42 + partition_index)
    sampled = random.sample(all_users, max(1, int(len(all_users) * SAMPLE_FRACTION)))
    logging.info("Partition %s: sampled %d / %d users", label, len(sampled), len(all_users))

    movie_embs = _load_movie_embeddings()
    emb_dim = next(iter(movie_embs.values())).shape[0]

    emb_rows, uid_list = [], []
    for uid in sampled:
        movie_ids = cumulative[cumulative["user_id"] == uid]["movie_id"].values
        vecs = [movie_embs[mid] for mid in movie_ids if mid in movie_embs]
        vec = np.mean(np.stack(vecs), axis=0).astype(np.float32) if vecs else np.zeros(emb_dim, dtype=np.float32)
        emb_rows.append(vec)
        uid_list.append(uid)

    matrix = np.stack(emb_rows)
    _save_npy(matrix, f"{USER_EMBEDDINGS_PREFIX}/{label}/embeddings.npy")
    _save_npy(np.array(uid_list, dtype=np.int32), f"{USER_EMBEDDINGS_PREFIX}/{label}/user_ids.npy")

    summary = pd.DataFrame({
        "user_id": uid_list,
        "partition": label,
        "is_new_user": [uid in new_users for uid in uid_list],
        "num_ratings": [int(len(cumulative[cumulative["user_id"] == uid])) for uid in uid_list],
    })
    boto3.client("s3").put_object(
        Bucket=S3_BUCKET,
        Key=f"{USER_EMBEDDINGS_PREFIX}/{label}/user_summary.csv",
        Body=summary.to_csv(index=False).encode("utf-8"),
    )
    logging.info("Partition %s complete. Embedding shape: %s", label, str(matrix.shape))


with DAG(
    dag_id="step3_user_embeddings",
    start_date=datetime(2026, 6, 6),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["step3", "hw4"],
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:

    t1 = PythonOperator(task_id="part1", python_callable=compute_embeddings, op_kwargs={"partition_index": 0})
    t2 = PythonOperator(task_id="part2", python_callable=compute_embeddings, op_kwargs={"partition_index": 1})
    t3 = PythonOperator(task_id="part3", python_callable=compute_embeddings, op_kwargs={"partition_index": 2})
    t4 = PythonOperator(task_id="part4", python_callable=compute_embeddings, op_kwargs={"partition_index": 3})

    t1 >> t2 >> t3 >> t4
