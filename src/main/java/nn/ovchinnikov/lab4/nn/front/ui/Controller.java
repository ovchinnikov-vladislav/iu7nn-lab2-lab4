package nn.ovchinnikov.lab4.nn.front.ui;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.stage.FileChooser;
import nn.ovchinnikov.lab4.nn.front.App;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

public class Controller {

    private static final Logger log = LoggerFactory.getLogger(Controller.class);

    private ComputationGraph network;
    private static final int HEIGHT = 150;
    private static final int WIDTH = 150;
    private static final int CHANNELS = 3;

    private static final String DEFAULT_INIT_DIRECTORY = "D:" + File.separator + "nn-network";

    private final FileChooser fileChooser = new FileChooser();

    @FXML
    private ImageView imageView;

    @FXML
    private TextField textAnimal;

    private String prevPathDirectory = DEFAULT_INIT_DIRECTORY;

    @FXML
    public void initialize() {
        imageView.setPreserveRatio(true);
        imageView.setFitHeight(500);
        try {
            network = ComputationGraph.load(new File("model/model.bin"), true);
        } catch (IOException exc) {
            exc.printStackTrace();
        }
    }

    @FXML
    public void selectedImageClicked(MouseEvent event) {
        textAnimal.setText(null);
        fileChooser.setInitialDirectory(new File(prevPathDirectory));
        File file = fileChooser.showOpenDialog(((Button) event.getSource()).getScene().getWindow());
        if (file != null) {
            prevPathDirectory = file.getParent();
            Image image = new Image(file.toURI().toString());
            imageView.setImage(image);
            recognized(file.getAbsolutePath());
        }
    }

    public void recognized(String pathFile) {
        if (pathFile != null) {
            try {
                NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
                INDArray image = loader.asMatrix(new FileInputStream(pathFile));
                DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
                scaler.transform(image);
                INDArray predictClasses = network.outputSingle(image);
                int searchClass = -1;
                double maxValue = 0;
                for (int i = 0; i < predictClasses.columns(); i++) {
                    if (maxValue <= predictClasses.getDouble(i)) {
                        searchClass = i;
                        maxValue = predictClasses.getDouble(i);
                    }
                }

                System.out.println(Arrays.toString(predictClasses.toDoubleVector()));
                switch (searchClass) {
                    case 0:
                        textAnimal.setText("It's CAT!");
                        break;
                    case 1:
                        textAnimal.setText("It's DOG!");
                        break;
                }
            } catch (IOException exc) {
                exc.printStackTrace();
            }
        }
    }

}
