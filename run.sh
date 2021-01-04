# Download kaggle token json and accept dataset terms and conditions
# otherwise we get auth errors.
kaggle competitions download -c ga-customer-revenue-prediction
hadoop fs -copyFromLocal  test_v2.csv /ga_customer_revenue_prediction_test_v2.csv
hadoop fs -copyFromLocal  train_v2.csv /ga_customer_revenue_prediction_train_v2.csv
spark-shell --conf spark.executor.memory=7G --conf spark.executor.memoryOverhead=1G   --conf spark.driver.memory=1G --packages com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc3
# execute dredge.scala on it then download the predictions
hadoop fs -copyToLocal /ga_customer_revenue_prediction_2017_10_01-09.csv ./
hadoop fs -copyToLocal /ga_customer_revenue_prediction_2017_10_10-19.csv ./
hadoop fs -copyToLocal /ga_customer_revenue_prediction_train_v2_new.lightbm.model ./
# Finally execute visualize.ipynb
