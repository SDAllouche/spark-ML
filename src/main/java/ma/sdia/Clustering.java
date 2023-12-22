package ma.sdia;

import org.apache.commons.math3.ml.clustering.evaluation.ClusterEvaluator;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.List;

public class Clustering {
    public static void main(String[] args) {
        SparkSession ss=SparkSession.builder().appName("Spark ML").master("local[*]").getOrCreate();

        Dataset<Row> dataset=ss.read().option("inferSchema", true).option("header",true).csv("Mall_Customers.csv");

        VectorAssembler vectorAssembler=new VectorAssembler().setInputCols(
                new String[]{"Age","Annual Income (k$)","Spending Score (1-100)"}
        ).setOutputCol("features");

        Dataset<Row> assembledDS = vectorAssembler.transform(dataset);
        MinMaxScaler scaler=new MinMaxScaler().setInputCol("features").setOutputCol("scaled_features");
        Dataset<Row> scaledFeatures=scaler.fit(assembledDS).transform(assembledDS);

        KMeans kMeans=new KMeans().setSeed(0).setK(5).setFeaturesCol("scaled_features").setPredictionCol("prediction");

        KMeansModel kMeansModel= kMeans.fit(scaledFeatures);
        Dataset<Row> prediction= kMeansModel.transform(scaledFeatures);
        prediction.show();

        ClusteringEvaluator evaluator=new ClusteringEvaluator();
        double score=evaluator.evaluate(prediction);
        System.out.println("Evaluating Score : "+score);

    }
}
