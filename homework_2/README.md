# Homework 2
Nick Visuthikosol

## How to Run
1. Open `homework_2.ipynb`
2. Run cells top to bottom

## Expected Outputs

### Task 1: Download and Upload Dataset
- `ml-1m.zip` uploaded to S3 at `homework_2/data/ml-1m.zip`
- Three dataframes loaded: `movies` (3,883 rows), `ratings` (1,000,209 rows), `users` (6,040 rows)

### Task 2: BERT Embeddings
- 887 movies filtered (pre-1980) from the full catalogue
- Each movie encoded using `distilbert-base-uncased` 
- Embeddings saved as `embeddings.pkl` and uploaded to S3 at `embeddings/embeddings.pkl`

### Task 3: Recommendations (pre-1980)
- **Cold user** (no history): 5 movie recommendations
- **Top user** (top 5% by rating count): 5 recommendations based on last 3 watched movies
- Results uploaded to S3:
  - `recommendations/pre1980/cold_user.csv`
  - `recommendations/pre1980/top_user.csv`

### Task 4: Full Dataset
- Same pipeline as Task 2 and 3 but on all 3,883 movies
- Full embeddings saved as `full_embeddings.pkl` and uploaded to S3 at `embeddings/full_embeddings.pkl`
- Results uploaded to S3:
  - `recommendations/full/cold_user.csv`
  - `recommendations/full/top_user.csv`

### Task 5: Personal Recommendations
- 10 personal movie ratings provided in `MY_RATINGS`
- Profile built by concatenating rated movie texts 
- 5 recommendations generated and uploaded to S3:
  - `user_profile/my_ratings.csv`
  - `recommendations/full/my_recommendations.csv`


## AI Usage

**Tool used:** Claude.ai

**Key prompts:**
- Debugging `NoCredentialsError` when connecting to AWS SSO from a local Mac
- Help with downloading the MovieLens dataset and uploading to S3
- Fixing `faiss-cpu` installation error
- Understanding how to encode text with distilBERT 
- Debugging S3 upload/download pattern for pickle files across Tasks 2, 3, 4, and 5

**What I changed and verified:**
- Chose `distilbert-base-uncased` to match the class example 
- Decided to use `faiss.IndexFlatIP` with normalized vectors 
- Chose the cold user strategy and top user strategy based on the class BERT example
- Verified Task 1 by checking S3 bucket contents after upload
- Verified Task 2 by confirming embedding matrix shape `(887, 768)` for pre-1980 and `(3883, 768)` for full
- Verified Tasks 3-5 by reviewing the printed recommendation DataFrames and confirming CSV files appeared in S3
