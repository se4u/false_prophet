import Array.range
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import com.microsoft.ml.spark.lightgbm.{LightGBMRegressionModel, LightGBMRegressor, LightGBMUtils}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer}
import org.apache.spark.ml.linalg.Vector
spark.sqlContext.udf.register("vector_to_array", (v: Any) => v.asInstanceOf[Vector].toArray)

def preprocess(df: DataFrame): DataFrame = {
    val tmp = df.selectExpr(
        "cast(nvl(from_json(totals, 'array<struct<transactionRevenue: string>>')[0]['transactionRevenue'], '0') as float) as transactionRevenue",
        "fullVisitorId",
        "channelGrouping",
        "socialEngagementType",
        "from_json(customDimensions, 'array<struct<value:string>>')[0]['value'] as customDimensionsValue",
        "from_json(device, 'struct<browser:string, operatingSystem:string, deviceCategory: string>') as _device",
        "from_json(geoNetwork, 'struct<continent: string, country: string>') as _geoNetwork",
        "from_json(trafficSource, 'array<struct<referralPath: string, campaign: string, source: string>>')[0] as _trafficSource")

    tmp.selectExpr(
      "transactionRevenue",
      "fullVisitorId",
      "nvl(channelGrouping, 'null') as channelGrouping",
      "nvl(socialEngagementType, 'null') as socialEngagementType",
      "nvl(customDimensionsValue, 'null') as customDimensions",
      "nvl(_device['browser'], 'null') as browser",
      "nvl(_device['operatingSystem'], 'null') as operatingSystem",
      "nvl(_device['deviceCategory'], 'null') as deviceCategory",
      "nvl(_geoNetwork['continent'], 'null') as continent",
      "nvl(_geoNetwork['country'], 'null') as country",
      "nvl(_trafficSource['referralPath'], 'null') as referralPath",
      "nvl(_trafficSource['campaign'], 'null') as campaign",
      "nvl(_trafficSource['source'], 'null') as source")      
}
val df = (spark.read.option("header", true).option("escape", "\"")
          .csv("/ga_customer_revenue_prediction_train_v2.csv"))
println(df.schema)
val old_period = Set("20171001","20171002","20171003","20171004","20171005",
                     "20171006","20171007","20171008","20171009")
val new_period = Set("20171011","20171012","20171013","20171014","20171015","20171016",
                     "20171017","20171018","20171019")
val columns = Seq(
    "channelGrouping", "socialEngagementType", "customDimensions", "browser", "operatingSystem", 
    "deviceCategory", "continent", "country", "referralPath", "campaign", "source")

val df_old_a = preprocess(df.filter($"date".isin(old_period.toSeq: _*))).cache()
val df_new_a = preprocess(df.filter($"date".isin(new_period.toSeq: _*))).cache()
println(s"old count: ${df_old_a.count()}, new count: ${df_new_a.count()}") 
// 34762, 27530

val old_mean = df_old_a.selectExpr("mean(transactionRevenue)").head(1)(0).getDouble(0)
val new_mean = df_new_a.selectExpr("mean(transactionRevenue)").head(1)(0).getDouble(0)
println(s"old transactionRevenue mean: ${old_mean}, new mean ${new_mean}")
// 623250.9604740809, 991015.2555030875

val indexers = columns.map(
    col => 
    new StringIndexer().setInputCol(col).setOutputCol(col + "_index").setHandleInvalid("keep")  
)
val featurizer = new VectorAssembler().setInputCols(columns.map(_ + "_index").toArray).setOutputCol("features")
val pipeline = new Pipeline().setStages((indexers :+ featurizer).toArray).fit(df_old_a)

// Write all the string indexes to a file. TODO: figure out json in scala
reflect.io.File("string_indexes.txt").writeAll(
  pipeline.stages.slice(0, columns.size).map(x => 
    x.asInstanceOf[StringIndexerModel].getInputCol 
    + "\n" 
    + x.asInstanceOf[StringIndexerModel].labels.toList.mkString("\n"))
  .mkString("\n\n\n"))


val df_old = pipeline.transform(df_old_a)
val df_new = pipeline.transform(df_new_a)


// Build a lightGBM classifier on df_new, then execute inference on df_new and df_old !
// If all we care about is the descriptive capability of the model
// and we never show the predictive capability.
// then we can overfit with glee. Its plain-old "data-dredging" after all. 
// Just make sure no one is looking at the code.

val estimator = (new LightGBMRegressor()
      .setCategoricalSlotIndexes(range(0, columns.size))
      .setLabelCol("transactionRevenue")
      .setFeaturesCol("features")
      .setNumIterations(1)
      .setNumTasks(1)
      // .setBaggingFraction(0.75)
      // .setFeatureFraction(0.9)
      // .setLambdaL1(1.0)
      // .setLambdaL2(1.0)
      // .setLearningRate(1.0)
      // .setMaxBin(254)
      // .setMaxDepth(3)
      // .setMinDataInLeaf(200)
      // .setNumLeaves(100)
      .setObjective("regression_l1")
      // .setObjective("huber")
      .setLeafPredictionCol("leaf")
      // .setFeaturesShapCol("shap")
)
val paramGrid = (new ParamGridBuilder()
      .addGrid(estimator.numLeaves, Array(10, 25, 50, 100))
      .addGrid(estimator.learningRate, Array(1.0, 0.5, 0.25, 0.125, 0.05))
      .build())
val evaluator = new RegressionEvaluator().setLabelCol("transactionRevenue")
val trainValidationSplit = (new TrainValidationSplit()
      .setEstimator(estimator)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
      .setParallelism(4))
val model = trainValidationSplit.fit(df_old)
val pred_old = model.transform(df_old).selectExpr("transactionRevenue", "prediction", "vector_to_array(leaf)[0] as leaf")
val pred_new = model.transform(df_new).selectExpr("transactionRevenue", "prediction", "vector_to_array(leaf)[0] as leaf")
println(s"${evaluator.evaluate(pred_old)} ${evaluator.evaluate(pred_new)}")
model.bestModel.asInstanceOf[LightGBMRegressionModel].saveNativeModel(
  "/ga_customer_revenue_prediction_train_v2_new.lightbm.model", overwrite=true)
pred_old.repartition(1).write.mode("overwrite").option("header", true).csv(
  "/ga_customer_revenue_prediction_2017_10_01-09.csv")
pred_new.repartition(1).write.mode("overwrite").option("header", true).csv(
  "/ga_customer_revenue_prediction_2017_10_10-19.csv")