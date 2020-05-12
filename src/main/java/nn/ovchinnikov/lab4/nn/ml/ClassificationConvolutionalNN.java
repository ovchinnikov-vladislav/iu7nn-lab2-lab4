package nn.ovchinnikov.lab4.nn.ml;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitDistribution;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.FaceNetNN4Small2;
import org.deeplearning4j.zoo.model.InceptionResNetV1;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class ClassificationConvolutionalNN {

    private static final Logger log = LoggerFactory.getLogger(ClassificationConvolutionalNN.class);
    private static final int height = 150;
    private static final int width = 150;
    private static final int channels = 3;

    private static final long seed = 1234;
    private static final Random rng = new Random(seed);
    private static int epochs = 3;
    private static int batchSize = 50;

    private int numLabels;

//    private static final String dataLocalPath = "D:" + File.separator + "nn-network" + File.separator + "labs" + File.separator +
//            "datasets" + File.separator + "land-train";

     private static final String dataLocalPathLandTrain = "D:\\nn-network\\animals_trains";
    private static final String dataLocalPathLandTest = "D:\\nn-network\\animals_tests";
//    private static final String dataLocalPathLandTrain = "D:" + File.separator + "nn-network" + File.separator + "labs" + File.separator +
//            "datasets" + File.separator + "land-train";
//
//    private static final String dataLocalPathLandTest = "D:" + File.separator + "nn-network" + File.separator + "labs" + File.separator +
//            "datasets" + File.separator + "land-test";

    public static void main(String[] args) {
        try {
            new ClassificationConvolutionalNN().build();
        } catch (Exception exc) {
            log.error("Cause ", exc);
        }
    }

    public void build() throws Exception {
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File pathTrainsData = new File(dataLocalPathLandTrain);
        File pathTestsData = new File(dataLocalPathLandTest);
        FileSplit fileSplitTrain = new FileSplit(pathTrainsData, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit fileSplitTest = new FileSplit(pathTestsData, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = Math.toIntExact(fileSplitTrain.length());
        numLabels = Objects.requireNonNull(fileSplitTest.getRootDir().listFiles(File::isDirectory)).length;
        int maxPathsPerLabel = 18;
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, maxPathsPerLabel);

        double splitTrainTest = 100;
        InputSplit[] inputSplits = fileSplitTrain.sample(pathFilter, splitTrainTest, 100 - splitTrainTest);
        InputSplit trainData = inputSplits[0];
        inputSplits = fileSplitTrain.sample(pathFilter, splitTrainTest, 100 - splitTrainTest);
        InputSplit testData = inputSplits[0];

        System.out.println(fileSplitTrain.length());
        System.out.println(fileSplitTest.length());
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
        boolean shuffle = false;

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(new Pair<>(flipTransform1, 0.9),
                new Pair<>(flipTransform2, 0.8),
                new Pair<>(warpTransform, 0.5));

        ImageTransform transform = new PipelineImageTransform(pipeline, shuffle);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        ComputationGraph network = ResNet50.init(numLabels);

        learnStartByComputationGraph(network, fileSplitTrain, fileSplitTest, scaler, labelMaker, transform);
    }

    private void learnStartByComputationGraph(ComputationGraph network, InputSplit trainData,
                            InputSplit testData, DataNormalization scaler,
                            ParentPathLabelGenerator labelMaker, ImageTransform transform) throws IOException {

        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator trainIter;

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);

        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testData);


        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, numLabels);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1), new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));

        // Train without transformations
        trainRR.initialize(trainData, null);
        trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, numLabels);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        network.fit(trainIter, epochs);

        // Train with transformations
//        trainRR.initialize(trainData, transform);
//        trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, numLabels);
//        scaler.fit(trainIter);
//        trainIter.setPreProcessor(scaler);
//        network.fit(trainIter, epochs);

        testIter.reset();
        DataSet testDataSet = testIter.next();
        List<String> allClassLabels = trainRR.getLabels();
        int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
        INDArray predictedClasses = network.outputSingle(testDataSet.getFeatures());
        String expectedResult = allClassLabels.get(labelIndex);
        int searchClass = -1;
        double maxValue = 0;
        for (int i = 0; i < predictedClasses.columns(); i++) {
            if (maxValue <= predictedClasses.getDouble(i)) {
                searchClass = i;
                maxValue = predictedClasses.getDouble(i);
            }
        }
        String modelPrediction = allClassLabels.get(searchClass);
        System.out.println("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");

        System.out.println("Save model....");
        network.save(new File("model/model.bin"));
        File newFile = new File("model/class_label_name.txt");
        FileWriter writer = new FileWriter(newFile, false);
        int i = 0;
        for (String classLabels : allClassLabels) {
            writer.write(classLabels + " = " + i + "\n");
            i++;
        }
        writer.close();
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}).name(name).nOut(out).biasInit(bias).build();
    }

    @SuppressWarnings("SameParameterValue")
    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5, 5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2, 2}).name(name).build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).weightInit(new WeightInitDistribution(dist)).build();
    }

    private MultiLayerNetwork lenetNetwork() {
        // LeNet Model
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta())
                .list()
                .layer(0, convInit("CNN 1", channels, 50, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("MAX POOl 1", new int[]{2, 2}))
                .layer(2, conv5x5("CNN 2 (5x5)", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("MAX POOL 2", new int[]{2, 2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);
    }

    private MultiLayerNetwork alexnetModel() {
        /*
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         */

        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new AdaDelta())
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .l2(5 * 1e-4)
                .list()
                .layer(convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(maxPool("maxpool1", new int[]{3, 3}))
                .layer(conv5x5("cnn2", 256, new int[]{1, 1}, new int[]{2, 2}, nonZeroBias))
                .layer(new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(maxPool("maxpool2", new int[]{3, 3}))
                .layer(conv3x3("cnn3", 384, 0))
                .layer(conv3x3("cnn4", 384, nonZeroBias))
                .layer(conv3x3("cnn5", 256, nonZeroBias))
                .layer(maxPool("maxpool3", new int[]{3, 3}))
                .layer(fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new NormalDistribution(0, 0.005)))
                .layer(fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new NormalDistribution(0, 0.005)))
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }
}
