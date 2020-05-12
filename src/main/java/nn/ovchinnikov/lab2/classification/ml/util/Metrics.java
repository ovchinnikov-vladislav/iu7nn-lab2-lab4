package nn.ovchinnikov.lab2.classification.ml.util;

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.*;

public class Metrics {

    private static final Map<String, Map<String, Double>> modelMetrics = new TreeMap<>();

    public static void computeAndPrint(Dataset<Row> predictions, String model) {
        Map<String, Double> map = new TreeMap<>();

        System.out.println(model + ":");
        List<Row> rows = predictions.select("indexedLabel", "label").distinct().collectAsList();
        for (Row r : rows) {
            double d = r.getAs("indexedLabel");
            String label = r.getAs("label");
            double precision = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel")
                    .setMetricLabel(d).setPredictionCol("prediction").setMetricName("precisionByLabel").evaluate(predictions);
            double recall = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel")
                    .setMetricLabel(d).setPredictionCol("prediction").setMetricName("recallByLabel").evaluate(predictions);
            double fMeasure = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel")
                    .setMetricLabel(d).setPredictionCol("prediction").setMetricName("fMeasureByLabel").evaluate(predictions);
            map.put(label + " precision", precision);
            map.put(label + " recall", recall);
            map.put(label + " fMeasure", fMeasure);
            System.out.print("\t"+label + ":\tprecision = " + precision);
            System.out.print("\trecall = " + recall);
            System.out.println("\tf_measure = " + fMeasure);
        }
        double weightedFMeasure = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel")
                .setMetricName("weightedFMeasure").evaluate(predictions);
        map.put("Weighted FMeasure", weightedFMeasure);
        System.out.println("\tWeighted FMeasure = " + weightedFMeasure +"\n");

        modelMetrics.put(model, Collections.unmodifiableMap(map));
    }

    public static Map<String, Map<String, Double>> getAllComputedModelMetrics() {
        return Collections.unmodifiableMap(modelMetrics);
    }
}
