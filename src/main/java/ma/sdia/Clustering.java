package ma.sdia;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Clustering {
    public static void main(String[] args) {
        SparkSession ss=SparkSession.builder().appName("Spark ML").master("local[*]").getOrCreate();

        Dataset<Row> dataset=ss.read().option("inferSchema", true).option("header",true).csv("Mall_Customers.csv");

        VectorAssembler vectorAssembler=new VectorAssembler().setInputCols(
                new String[]{"Age","Annual Income (k$)","Spending Score (1-100)"}
        ).setOutputCol("Features");

        Dataset<Row> assembledDS = vectorAssembler.transform(dataset);

        KMeans kMeans=new KMeans().setK(3).setFeaturesCol("Features").setPredictionCol("cluster");

        KMeansModel kMeansModel= kMeans.fit(assembledDS);
        Dataset<Row> prediction= kMeansModel.transform(assembledDS);
        prediction.show();
    }
}
