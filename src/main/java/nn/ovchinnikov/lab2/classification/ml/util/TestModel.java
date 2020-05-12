package nn.ovchinnikov.lab2.classification.ml.util;

import nn.ovchinnikov.lab2.classification.util.VoiceDataReader;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class TestModel {

    public static void test(Dataset<Row> testData, PipelineModel model, String nameModel) {

        Dataset<Row> predictions = model.transform(testData);

       // predictions.select("predictedLabel", "label").show(100);

        Metrics.computeAndPrint(predictions, nameModel);
    }
}
