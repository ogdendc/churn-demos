# Databricks notebook source
slide_id = '1jeg7B1I7d0d8CBImOUbMF7M3794OXYUfdFxAoWtAJ4M'
slide_number = '1'

displayHTML(f'''

<iframe
  src="https://docs.google.com/presentation/d/{slide_id}/embed?slide={slide_number}&rm=minimal"
  frameborder="0"
  width="120%"
  height="500"
></iframe>

''')

# COMMAND ----------

# MAGIC %md
# MAGIC #&emsp;
# MAGIC #&emsp;
# MAGIC #&emsp;
# MAGIC #&emsp;
# MAGIC #Let's start with a quick end-to-end overview of:
# MAGIC >##navigating the Databricks platform
# MAGIC >##a Machine Learning project cycle *all within* Databricks
# MAGIC #&emsp;
# MAGIC >##(after which we'll dig-in a bit deeper)
# MAGIC #&emsp;
# MAGIC #&emsp;
# MAGIC #&emsp;

# COMMAND ----------

# MAGIC %md #Self-Service Infrastructure...
# MAGIC >##**Spinning-up your compute space...just a few clicks in one easy screen:**
# MAGIC # <--- 
# MAGIC <img src ='http://drive.google.com/uc?export=view&id=1zVLAJjTs1GwfuTHo5xa9mFyW--AxTLjZ' width="500" height="1000">

# COMMAND ----------

# MAGIC %md 
# MAGIC >##**Accessing your data...just a few clicks in one easy screen:**
# MAGIC # <--- 
# MAGIC <img src ='http://drive.google.com/uc?export=view&id=1SswquXLET79EKU-PE8Vng9Q9m3zFI9Kt' width="1000" height="1000">

# COMMAND ----------

# This creates the "path_name" field displayed at the top of the notebook.
# Enter a value to be used for creating a unique path for local files, DBFS, and database creation

dbutils.widgets.text("path_name", "enter a path name");

# COMMAND ----------

# calling some setup code store in the same folder as this notebook

path_name = dbutils.widgets.get("path_name")

setup_responses = dbutils.notebook.run("./setup_stuff/setup_process", 0, {"path_name": path_name}).split()

local_data_path = setup_responses[0]
dbfs_data_path = setup_responses[1]
database_name = setup_responses[2]

print(f"Path to be used for Local Files: {local_data_path}")
print(f"Path to be used for DBFS Files: {dbfs_data_path}")
print(f"Database Name: {database_name}")

# COMMAND ----------

# Let's set the default database name so we don't have to specify it on every query

spark.sql(f"USE {database_name}")

# if we did not use the above, and used the code below as-is, the delta tables would get stored in 
#     dbfs:/user/hive/warehouse/  rather than dbfs:/user/hive/warehouse/{database_name}.db/

# COMMAND ----------

# MAGIC %md #Easy Collaboration...
# MAGIC >##Users swiching back-and-forth to programming language of their choice (Python, R, Scala, SQL)
# MAGIC >##Allowing Data Engineers, Data Scientists and Business Analysts to collaborate in the same environment.

# COMMAND ----------

# MAGIC %md 
# MAGIC # 
# MAGIC # 
# MAGIC # Enter the Data Engineer...

# COMMAND ----------

# Read the downloaded historical data into a dataframe

dataPath = f"dbfs:/FileStore/{path_name}/churn_modeling.csv"
spark_df = spark.read.csv(dataPath, header=True, inferSchema=True)  # letting Spark infer the schema
display(spark_df)

# COMMAND ----------

# API to present Spark dataframe as a queriable view, so we can use spark.sql() or run %sql cells:

spark_df.createOrReplaceTempView("spark_view")

# COMMAND ----------

# MAGIC %md #Data Engineer is prepping the data for our Data Scientist
# MAGIC >## switching between SQL and Python as-desired
# MAGIC >## analyzing and transforming the data into a "silver medallion" table to add to the database

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT format_number(count(DISTINCT cust_num), 0)     AS customer_cnt,
# MAGIC        format_number(avg(vol_disco_ind), '0.0%')      AS overall_churn_rate
# MAGIC FROM spark_view

# COMMAND ----------

# MAGIC %md ### Validating and exploring...

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC    geo_demog_desc,
# MAGIC    count(cust_num) AS cust_cnt,
# MAGIC    avg(vol_disco_ind) AS churn_rate
# MAGIC FROM spark_view
# MAGIC GROUP BY geo_demog_desc

# COMMAND ----------

print (spark_df.dtypes)

# COMMAND ----------

# MAGIC %md #Data Scientist collaborates with Data Engineer throughout
# MAGIC >## our Data Scientist engages in the Analysis
# MAGIC >## investigating distributions, missing values, etc
# MAGIC >## using the open source libraries they're familiar with (pandas, numpy, matplotlib, etc)
# MAGIC >## collaboratively building the notebook together...
# MAGIC >## both switching between SQL and Python as-desired

# COMMAND ----------

import pandas as pd

pandas_df = spark_df.toPandas()

pandas_df.shape

# COMMAND ----------

print (pandas_df.dtypes)

# COMMAND ----------

import numpy as np

# setting some display options
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 5)

print(" summary stats for churn data ") 
print(" ")

# describe only the numeric variables:
print(pandas_df.describe(include = [np.number]).transpose())
print(" ")

# followed by only the character variables:
print(pandas_df.describe(include = [np.object]).transpose())
print(" ")

# COMMAND ----------

# analyzing the distribution of the tenure variable:
print("percentiles for tenure")
print(" ")
pandas_df.tenure_months.quantile([0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00])

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib

our_plot = pandas_df['tenure_months'].plot(kind='hist', bins=72, figsize=(15,7), color="green", fontsize=15,
                                         xticks=[0, 24, 48, 72, 96, 120, 144], xlim=[0,144]);

our_plot.set_title("Customer Tenure Distribution", fontsize=32)
our_plot.set_xlabel("Tenure in Months", fontsize=26)
our_plot.set_ylabel("Count of Accounts", fontsize=26)
plt.show()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT cpe_cnt
# MAGIC FROM spark_view

# COMMAND ----------

print(" ")
print("Count of Missing Values")
pandas_df.isnull().sum()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT format_number(avg(credit_score), '0') AS avg_credit_score,
# MAGIC        format_number(percentile(cpe_cnt,0.5), '0') AS median_cpe
# MAGIC FROM spark_view

# COMMAND ----------

# imputing missing and invalid values

# the fillna function provides minimal coding to impute missing values:
pandas_df['prod_flg_02'] = pandas_df['prod_flg_02'].fillna(0) 
pandas_df['credit_score'] = pandas_df['credit_score'].fillna(938) 

# when cpe_cnt = -1 then replace with median value
pandas_df.loc[(pandas_df['cpe_cnt']==-1),'cpe_cnt'] = 2

# dropping the comment_ind column since this is always a value of 1 and adds no information
pandas_df = pandas_df.drop(['comment_ind'],axis=1)

# COMMAND ----------

print(" ")
print("Count of Missing Values")
pandas_df.isnull().sum()

# COMMAND ----------

# some feature engineering
pandas_df["prod_flg_tot"]=1+pandas_df["prod_flg_01"]+pandas_df["prod_flg_02"]+pandas_df["prod_flg_03"]+pandas_df["prod_flg_04"]+pandas_df["prod_flg_05"]+pandas_df["prod_flg_06"]+pandas_df["prod_flg_07"]+pandas_df["prod_flg_08"]+pandas_df["prod_flg_09"]

print("percentiles for total product count")
print(" ")
pandas_df.prod_flg_tot.quantile([0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00])

# COMMAND ----------

# this cleaned "bronze" table to be written to Delta

# convert pandas DF to Spark DF
spark_df = spark.createDataFrame(pandas_df)

# Create a temporary view on the dataframes to enable SQL
spark_df.createOrReplaceTempView("churn_bronze")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a Delta Lake table for the churn bronze table
# MAGIC 
# MAGIC DROP TABLE IF EXISTS churn_modeling_bronze;
# MAGIC 
# MAGIC CREATE TABLE churn_modeling_bronze
# MAGIC USING delta
# MAGIC AS SELECT * FROM churn_bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM churn_modeling_bronze
# MAGIC WHERE prod_flg_tot = 10

# COMMAND ----------

# MAGIC %sql 
# MAGIC OPTIMIZE churn_modeling_bronze ZORDER BY DMA

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM churn_modeling_bronze
# MAGIC WHERE prod_flg_tot = 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a Delta Lake table for the churn bronze table, partitioned by geo_demog_desc
# MAGIC 
# MAGIC -- DROP TABLE IF EXISTS churn_modeling_bronze_partitioned;
# MAGIC 
# MAGIC -- CREATE TABLE churn_modeling_bronze_partitioned
# MAGIC -- USING delta
# MAGIC -- PARTITIONED BY (geo_demog_desc)
# MAGIC -- AS SELECT * FROM churn_bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC -- SELECT *
# MAGIC -- FROM churn_modeling_bronze_partitioned
# MAGIC -- WHERE prod_flg_tot = 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DESCRIBE EXTENDED churn_modeling_bronze_partitioned

# COMMAND ----------

# dbutils.fs.ls(f"dbfs:/user/hive/warehouse/{database_name}.db/churn_modeling_bronze_partitioned")

# COMMAND ----------

spark_df = spark.read.table("churn_modeling_bronze")
display(spark_df.summary())

# COMMAND ----------

# MAGIC %md #Data Engineer performs one last step
# MAGIC >##encoding categorical/nominal variables to fascilitate Machine Learning models
# MAGIC >##with final modifications stored in a "silver" version of modeling data

# COMMAND ----------

# for xgboost modeling, need to do some one-hot encoding
#    DMA, geo_demog_desc, dc_flg, contract_flg, acct_type, product

from pyspark.sql.functions import col
import re

encoded_df = spark_df

# Spark-based one-hot encoding (results in true/false boolean indicators)

for column in ['DMA',
               'geo_demog_desc',
               'dc_flg',
               'contract_flg', 
               'acct_type', 
               'product'
              ]:
  
  levels = sorted(map(lambda r: r[column], encoded_df.select(column).distinct().collect()))
  
  for value in levels:
    # modifying values in columns to remove dashes and blanks, since these values would be included in column names
    value=value.replace(' ', '')
    value=value.replace('-', '')
    value=value.replace(',', '')
    encoded_df = encoded_df.withColumn(column+"_"+value, col(column) == value)
  
  encoded_df = encoded_df.drop(column)

# COMMAND ----------

display(encoded_df)

# COMMAND ----------

# Create a temporary view on the dataframes to enable SQL
encoded_df.createOrReplaceTempView("churn_silver")

# COMMAND ----------

# MAGIC %md ##Creating a "silver" Delta Lake table, cleaned, featurized, one-hot encoded, and ready for churn modeling:

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS churn_modeling_silver;
# MAGIC 
# MAGIC CREATE TABLE churn_modeling_silver
# MAGIC USING delta
# MAGIC AS SELECT * FROM churn_silver

# COMMAND ----------

# MAGIC %md #Data Scientist grabs the silver table, and starts to model using the power of MLFlow

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='http://drive.google.com/uc?export=view&id=1sFQerCoFWFzSttFEZ-0troSpYQAFehEa' width="1000" height="1400">

# COMMAND ----------

# enable MLflow autologging for Spark data sources now, before they're used; this will be explained later.

import mlflow.spark

mlflow.spark.autolog()

# COMMAND ----------

spark_df = spark.read.table("churn_modeling_silver")

# COMMAND ----------

# Randomly split data into training and test sets. Set seed for reproducibility
    
(xtrain_spark_df, xtest_spark_df) = spark_df.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

ytrain_spark_df = xtrain_spark_df.select('vol_disco_ind')
xtrain_spark_df = xtrain_spark_df.drop('vol_disco_ind')

ytest_spark_df = xtest_spark_df.select('vol_disco_ind')
xtest_spark_df = xtest_spark_df.drop('vol_disco_ind')

# COMMAND ----------

print("X training rows = ", xtrain_spark_df.count(), "    X training columns = ", len(xtrain_spark_df.columns))
print("Y training rows = ", ytrain_spark_df.count(), "    Y training columns = ", len(ytrain_spark_df.columns))

# COMMAND ----------

import xgboost as xgb

ytrain_pd_df = ytrain_spark_df.toPandas()  # Note:  toPandas for converting spark df, but to_pandas to convert koalas to pandas
xtrain_pd_df = xtrain_spark_df.toPandas()
ytest_pd_df  = ytest_spark_df.toPandas()
xtest_pd_df  = xtest_spark_df.toPandas()

# use DMatrix for xgboost
dtrain = xgb.DMatrix(xtrain_pd_df, label=ytrain_pd_df)
dtest  = xgb.DMatrix(xtest_pd_df,  label=ytest_pd_df)

# COMMAND ----------

# MAGIC %md #Data Scientist trains an XGBoost model
# MAGIC > ##using the same *hyperparameters* they used last year
# MAGIC > ##the result is a mediocre fit

# COMMAND ----------

# training the model

param = {
    'objective':                'binary:logistic',
    'num_class':                1,                  # the number of classes that exist in this datset 
                                                    # (set to 1 for binary:logistic, set to 2 for multi:softprob)
    'eval_metric':              'auc',              # auc is good for binary.  also error for misclassification, 
                                                    # error%t to specify cutoff other than 0.5
    'seed':                     42,                 # specify for repeatability of results
    'verbosity':                0,                  # 0=silent, 1=warnings, 2=info, 3=debug
    'max_depth':                5,
    'min_child_weight':         1,                  # 0+. default=1. The larger min_child_weight is, the more conservativ e 
                                                    # the algorithm will be (stopping leaf splits).
    'learning_rate':            0.2,                # [0,1] In each iteration, shrinks the feature weights to make the boosting process more conservative.
    'subsample':                1.0,                # ratio of training data randomly sampled in each iteration, default = 1 or 100%
    'colsample_bytree':         1.0,                # ratio of columns to be sampled in each tree/iteration, default = 1 or 100%
    'gamma':                    0,                  # default = 0. Minimum loss reduction required to make a further partition on a leaf node. 
                                                    # Larger = more conservative. No max?
    'lambda':                   1,                  # default = 1. L2 regularization term on weights. 
                                                    # Increasing this value will make model more conservative.  No max?
    'alpha':                    0,                  # default = 0. L1 regularization term on weights. 
                                                    # Increasing this value will make model more conservative.  No max?
    'booster':                  'gblinear'          # default is gbtree. dart & gbtree are tree-based. values can also be gblinear.  
    }

# number of boosting iterations
num_round=200

# if no improvement in eval_metric in X rounds then stop training
early_stop=50
  
bst = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round, evals=[(dtest, "test")], early_stopping_rounds=early_stop, verbose_eval=False)  #verbose_eval=False to suppress iterations in output

print("Best Iteration: {}".format(bst.attr('best_iteration')))
print("Best Score (based on chosen eval_metric): {}".format(bst.best_score))

# COMMAND ----------

# MAGIC %md ##Data Scientists wants to improve the predictive power...

# COMMAND ----------

# MAGIC %md #Data Scientist uses built-in Databricks ML capability:  *hyperopt* 
# MAGIC > ## to systematically search for *optimal hyperparameters*
# MAGIC > ## distributing this work across our cluster
# MAGIC > ## resulting in significant model fit improvement
# MAGIC > ## in much less time than traditional approaches to such tuning

# COMMAND ----------

from math import exp
import xgboost as xgb
from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope 
import mlflow
import numpy as np
      
def params_to_xgb(params):
  return {
    'objective':               'binary:logistic',
    'num_class':               1,                  # the number of classes that exist in this datset
    'eval_metric':             'auc',
    'seed':                    42,
    'verbosity':               0,                   # 0=silent, 1=warnings, 2=info, 3=debug
    'max_depth':               params['max_depth'],
    'learning_rate':           params['learning_rate'],
    'subsample':               params['subsample'],
    'colsample_bytree':        params['colsample_bytree'],
    'gamma':                   params['gamma'],
    'lambda':                  params['lambda'],
    'alpha':                   params['alpha'],
    'booster':                 params['booster']           
  } 

def train_model(params):
  dtrain = xgb.DMatrix(xtrain_pd_df, label=ytrain_pd_df)
  dtest  = xgb.DMatrix(xtest_pd_df,  label=ytest_pd_df) 
  
  num_round=200
  early_stop=50
  
  booster = xgb.train(params=params_to_xgb(params), dtrain=dtrain, num_boost_round=num_round, evals=[(dtest, "test")], early_stopping_rounds=early_stop, verbose_eval=False)  

  mlflow.log_param('best_iteration', booster.attr('best_iteration'))
  return {'status': STATUS_OK, 'loss': -(booster.best_score), 'booster': booster.attributes()}  
  # notice negative sign on our best_score (above); since we want to maximize AUC we are minimizing -AUC

search_space = {
  'max_depth':            scope.int(hp.quniform('max_depth', 2, 10, 1)),   # NOT using randint since it assumes no ordinal relationship
  'min_child_weight':     hp.uniform('min_child_weight', 0.01, 3),  
  'learning_rate':        hp.uniform('learning_rate', 0.01, 0.20),
  'subsample':            hp.uniform('subsample', 0.50, 1.00),
  'colsample_bytree':     hp.uniform('colsample_bytree', 0.50, 1.00),
  'gamma':                hp.uniform('gamma', 0.0, 0.5),
  'lambda':               hp.uniform('lambda', 1, 5),
  'alpha':                hp.uniform('alpha', 0.0, 0.5),  
  'booster':              hp.choice('booster', ['gblinear','gbtree'])  # including 'dart' is resource intensive
}

# configure hyperopt settings to distribute to all executors on workers
spark_trials = SparkTrials(parallelism=2)

# algo=tpe.suggest utilizes a Bayesian approach that iteratively and adaptively selects new hyperparameter settings to explore based on previous results
# algo=rand.suggest is a random a non-adaptive approach that samples over the search space
best_params = fmin(fn=train_model, space=search_space, algo=tpe.suggest, max_evals=100, trials=spark_trials, rstate=np.random.RandomState(123))

# COMMAND ----------

# MAGIC %md #Data Scientist sees a significant improvement in model fit...
# MAGIC >## and looks at the "Experiment" UI (see upper right of our notebook)
# MAGIC >## to get more info about the optimized hyperparameter search results

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='http://drive.google.com/uc?export=view&id=1CI_CDq_NqGZGNB3VCoN1SN511D6wNJz0' width="1200" height="1800">

# COMMAND ----------

# MAGIC %md ##...and after analyzing the hyperopt search results, can recall the optimal hyperparameters:

# COMMAND ----------

from hyperopt import space_eval
# print optimum hyperparameter settings
print(space_eval(search_space, best_params))

# COMMAND ----------

# MAGIC %md ##With a best set of hyperparameters chosen...
# MAGIC >### the final model is re-fit and logged with MLflow...
# MAGIC >###along with an analysis of feature importance from shap:

# COMMAND ----------

import mlflow
import mlflow.xgboost
import shap
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature

plt.close()

with mlflow.start_run() as run:
  X=xtrain_pd_df
  y=ytrain_pd_df
  best_iteration = int(spark_trials.best_trial['result']['booster']['best_iteration'])
  booster = xgb.train(params=space_eval(search_space, best_params), dtrain=xgb.DMatrix(X, label=y), num_boost_round=best_iteration)
  mlflow.log_params(best_params)
  mlflow.log_param('best_iteration', best_iteration)
  mlflow.xgboost.log_model(booster, "xgboost", input_example=X.head(), signature=infer_signature(X, y))

  shap_values = shap.TreeExplainer(booster).shap_values(X, y=y)
  shap.summary_plot(shap_values, X, plot_size=(14,6), max_display=10, show=False)
  plt.savefig("summary_plot.png", bbox_inches="tight")
  plt.close()
  mlflow.log_artifact("summary_plot.png")
  
  best_run = run.info

# COMMAND ----------

# MAGIC %md ##This model can then be registered as the current candidate model
# MAGIC >###assign to a "Staging" status for further evaluation in the Model Registry

# COMMAND ----------

import time

model_name = "churn_pred_v1"
client = mlflow.tracking.MlflowClient()
try:
  client.create_registered_model(model_name)
except Exception as e:
  pass

model_version = client.create_model_version(model_name, f"{best_run.artifact_uri}/xgboost", best_run.run_id)

time.sleep(5) # Just to make sure it's had a second to register
client.transition_model_version_stage(model_name, model_version.version, stage="Staging")

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='http://drive.google.com/uc?export=view&id=1UGRMjbSjCmJMqosLA1mps6gDO3kK1sC7' width="1200" height="1800">

# COMMAND ----------

# MAGIC %md
# MAGIC #Is this model as good as it gets?
# MAGIC >## Data Scientists and ML Engineer evaluate and scrutinize this Staged model.
# MAGIC >## One Data Scientist wants to use AutoML to benchmark this model's performance...

# COMMAND ----------

# MAGIC %md #Data Scientist uses Databricks AutoML
# MAGIC >## A no-code solution to automatically producing optimized machine learning results

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='http://drive.google.com/uc?export=view&id=18mV0e7briCqMhThG0RbMAQGHbflHtamO' width="600" height="900">

# COMMAND ----------

# MAGIC %md #AutoML results are automatically saved
# MAGIC >## A "glass box" methodology providing insights and reproducibility
# MAGIC > ###https://adb-984752964297111.11.azuredatabricks.net/?o=984752964297111#mlflow/experiments/4340760541715444

# COMMAND ----------

# MAGIC %md #ML Engineer sees the Staged model is as good as benchmark performance from AutoML
# MAGIC ># and deems to move the Staged model to Production

# COMMAND ----------

# run this after the Staged model is validated, to move to Production

client = mlflow.tracking.MlflowClient()
latest_model_detail = client.get_latest_versions("churn_pred_v1", stages=['Staging'])[0] # most recent staged version
client.transition_model_version_stage(model_name, latest_model_detail.version, stage="Production")

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='http://drive.google.com/uc?export=view&id=1OmcmdtZDmui1Im1UhQXeoJG9WNx0AZhs' width="1200" height="1800">

# COMMAND ----------

# MAGIC %md #ML Engineer wants to productionize the model and use for inference
# MAGIC >## could do a batch job
# MAGIC >## but in our example, we're going to use it for real-time scoring

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='http://drive.google.com/uc?export=view&id=1NMKFFli88qghUOipNi26Yya32W-qT4U8' width="600" height="900">

# COMMAND ----------

# MAGIC %md
# MAGIC ##Saving a JSON representation of an input record
# MAGIC >##to use in simulating real-time scoring

# COMMAND ----------

print(X.head(1).to_json(orient='records'))

# COMMAND ----------

# MAGIC %md #Easy interface to aid in serving an endpoint API to score in real-time
# MAGIC <img src ='http://drive.google.com/uc?export=view&id=1z9YgYx_ZFYRnzTmi-KM5W1OOlhyCtu87' width="1000" height="1500">

# COMMAND ----------

slide_id = '1x1q3z5HuJf5F2PJd0QcK4sGcKRJWKRqXQYC5vxRfiBI'
slide_number = '1'

displayHTML(f'''

<iframe
  src="https://docs.google.com/presentation/d/{slide_id}/embed?slide={slide_number}&rm=minimal"
  frameborder="0"
  width="100%"
  height="400"
></iframe>

''')
