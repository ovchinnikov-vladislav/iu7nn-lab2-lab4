package nn.ovchinnikov.lab2.classification.util;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

public class VoiceDataReader {

    public static JavaRDD<LabeledPoint> readJavaRDD(SparkSession spark, String csvFile) {

        Dataset<Row> data = spark.read()
                .option("header", true)
                .schema(getSchema())
                .csv(csvFile);

        return data.toJavaRDD().map((Function<Row, LabeledPoint>) v1 -> {
            String sex = v1.getAs("label");
            int s;
            if (sex.equals("male"))
                s = 0;
            else
                s = 1;
            Vector vector = Vectors.dense(
                    v1.getAs("meanfreq"),
                    v1.getAs("sd"),
                    v1.getAs("median"),
                    v1.getAs("Q25"),
                    v1.getAs("Q75"),
                    v1.getAs("IQR"),
                    v1.getAs("skew"),
                    v1.getAs("kurt"),
                    v1.getAs("sp_ent"),
                    v1.getAs("sfm"),
                    v1.getAs("mode"),
                    v1.getAs("centroid"),
                    v1.getAs("meanfun"),
                    v1.getAs("minfun"),
                    v1.getAs("maxfun"),
                    v1.getAs("meandom"),
                    v1.getAs("mindom"),
                    v1.getAs("maxdom"),
                    v1.getAs("dfrange"),
                    v1.getAs("modindx"));
            return new LabeledPoint(s, vector);
        });
    }

    public static Dataset<Row> readDataset(SparkSession spark, String csvFile) {
        Dataset<Row> dataset = spark.read()
                .option("header", true)
                .schema(getSchema())
                .csv(csvFile);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt", "sp_ent",
                        "sfm", "mode", "centroid", "meanfun", "minfun", "maxfun", "meandom", "mindom", "maxdom",
                        "dfrange", "modindx"
                }).setOutputCol("assemblerFeatures");

        dataset = assembler.transform(dataset);

        Normalizer normalizer = new Normalizer()
                .setInputCol("assemblerFeatures")
                .setOutputCol("features");

        dataset = normalizer.transform(dataset);

        dataset.show();

        return dataset;
    }

    private static StructType getSchema() {
        return new StructType()
                .add("meanfreq", "double")
                .add("sd", "double")
                .add("median", "double")
                .add("Q25", "double")
                .add("Q75", "double")
                .add("IQR", "double")
                .add("skew", "double")
                .add("kurt", "double")
                .add("sp_ent", "double")
                .add("sfm", "double")
                .add("mode", "double")
                .add("centroid", "double")
                .add("meanfun", "double")
                .add("minfun", "double")
                .add("maxfun", "double")
                .add("meandom", "double")
                .add("mindom", "double")
                .add("maxdom", "double")
                .add("dfrange", "double")
                .add("modindx", "double")
                .add("label", "string");
    }
}
