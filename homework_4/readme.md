# Homework 4 - Movie Recommendation Pipeline on AWS MWAA
Rhema Phiri, Nick Visuthikosol, Maya Solomon

## Environment
- MWAA Environment: de300-airflow-phirisolomonvst-v2
- S3 Bucket: phirisolomonvst

## How to Run

Upload both DAG files to s3://phirisolomonvst/dags/

Run:
1. Trigger step3_user_embeddings 
2. Trigger step4_recommendations 

## Step 3 - User Embeddings
Splits ratings into 4 time partitions and computes an embedding for each user by averaging the BERT embeddings of movies they rated. Samples 30% of users per partition. Outputs `.npy` files and a summary CSV to S3.
 
## Step 4 - Recommendations
For each partition, generates 5 movie recommendations for two users:
- Cold user: no watch history, gets the 5 most popular movies
- Top user: picked from the top 5% most active users, gets recommendations based on cosine similarity between their embedding and all movie embeddings
Results saved as a CSV to s3://phirisolomonvst/ml-1m/recommendations/.
 
## Expected Outputs
ml-1m/user_embeddings/part1/embeddings.npy
ml-1m/user_embeddings/part1/user_ids.npy
ml-1m/user_embeddings/part1/user_summary.csv
(same for part2, part3, part4)
 
ml-1m/recommendations/part1/recommendations_<timestamp>.csv
(same for part2, part3, part4)


## Step 5 - Scheduling

Step 3 runs once manually. Step 4 is scheduled every 12 hours. Both DAGs stay active for 48 hours
