package nn.ovchinnikov.lab2.classification.ml;

import nn.ovchinnikov.lab2.classification.ml.util.TestModel;
import nn.ovchinnikov.lab2.classification.util.VoiceDataReader;
import nn.ovchinnikov.lab2.classification.util.UtilFile;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Predictor;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class StackingML {

    public static void build(Dataset<Row> trainingData, Dataset<Row> testData) throws IOException {
        UtilFile.recursiveDeleteModel("datasets/model/stacking-voice.model");

        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(trainingData);

        StackingClassifier classifier = new StackingClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("features")
                .setBaseLearners(new Predictor<?,?,?>[] {new DecisionTreeClassifier(), new RandomForestClassifier(), new LogisticRegression(), new LinearSVC()})
                .setStacker(new LogisticRegression());

        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labelsArray()[0]);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{labelIndexer, classifier, labelConverter});

        PipelineModel model = pipeline.fit(trainingData);

        TestModel.test(testData, model, "Stacking Model");

        StackingClassificationModel stackingClassificationModel =
                (StackingClassificationModel) (model.stages()[1]);

        stackingClassificationModel.save("datasets/model/stacking-voice.model");
    }

}
