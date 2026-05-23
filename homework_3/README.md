# HW3: NYC Taxi Analytics with PySpark on EC2

## How to Run

**1. SSH into EC2**

**2. Download data from S3**

**3. Launch Jupyter on EC2**

**4. Open homework3.ipynb**

**5. Run all cells top to bottom.** 
The notebook covers:
   - Parts 1–3: Spark setup and data loading
   - Part 4: Schema standardization (yellow/green column renaming + union)
   - Part 5: Data cleaning (null removal, distance/fare filters, duration filter)
   - Part 6: Analytics — Q1–Q5 and Q8 (Random Forest fare prediction)
   - Part 7: Save results locally then upload to S3

## Output
Results are saved locally to /home/ec2-user/outputs/ and uploaded to S3 under s3://de300-hw3-visuthikosol/nyc-taxi-assignment/.


## AI Usage

**Tool used:** Claude AI

**Key prompts:**
- Debugging PySpark errors like S3 FileSystem scheme error, Py4JJavaError
- Identifying correct PySpark imports during analysis (for example: from pyspark.sql.functions import avg, round as spark_round, from pyspark.sql.functions import hour)
- Refresher on Random Forest setup in PySpark MLlib such as VectorAssembler, RandomForestRegressor, Pipeline, and RegressionEvaluator for training RMSE and test RMSE
- General debugging of cell errors throughout the notebook

**What I changed and verified:**
- Reviewed all generated code and ran each cell myself on EC2 to verify correct output
- Instead of the direct S3 write approach I locally saved then CLI upload  after encountering the Hadoop-AWS connector error
- Verified model outputs (RMSE, feature importances, plot) matched expected behavior from lecture examples
