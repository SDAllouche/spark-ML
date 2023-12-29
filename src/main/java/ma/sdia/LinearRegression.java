package ma.sdia;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class LinearRegression {
    public static void main(String[] args) {
        SparkSession ss=SparkSession.builder().appName("Spark ML").master("local[*]").getOrCreate();

        Dataset<Row> dataset=ss.read().option("inferSchema", true).option("header",true).csv("advertising.csv");
        dataset.show();

        VectorAssembler vectorAssembler=new VectorAssembler().setInputCols(
                new String[]{"TV","Radio","Newspaper"}
        ).setOutputCol("Features");

        Dataset<Row> assembledDS = vectorAssembler.transform(dataset);
        Dataset<Row> splits[] = assembledDS.randomSplit(new double[]{0.8, 0.2}, 123);

        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        org.apache.spark.ml.regression.LinearRegression regression=new org.apache.spark.ml.regression.LinearRegression().setLabelCol("Sales").setFeaturesCol("Features");
        LinearRegressionModel model = regression.fit(train);
        Dataset<Row> predictions = model.transform(test);
        predictions.show();
        System.out.println("Intercept="+model.intercept()+" coefficienrs="+model.coefficients());

    }
}