package nn.ovchinnikov.lab2.classification.ml;

import nn.ovchinnikov.lab2.classification.ml.util.TestModel;
import nn.ovchinnikov.lab2.classification.util.VoiceDataReader;
import nn.ovchinnikov.lab2.classification.util.UtilFile;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class LinearSVCML {

    public static void build(Dataset<Row> trainingData, Dataset<Row> testData) throws IOException {
        UtilFile.recursiveDeleteModel("datasets/model/svm-voice.model");

        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(trainingData);

        LinearSVC linearSVC = new LinearSVC()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("features");

        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labelsArray()[0]);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{labelIndexer, linearSVC, labelConverter});

        PipelineModel model = pipeline.fit(trainingData);

        TestModel.test(testData, model, "Linear SVC Model");

        LinearSVCModel linearSVCModel =
                (LinearSVCModel) (model.stages()[1]);

        linearSVCModel.save("datasets/model/svm-voice.model");
    }

}
