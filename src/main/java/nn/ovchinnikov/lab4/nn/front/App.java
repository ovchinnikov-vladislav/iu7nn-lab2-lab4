package nn.ovchinnikov.lab4.nn.front;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

/**
 * Created by klevis.ramo on 1/1/2018.
 */
public class App extends Application {

    @Override
    public void start(Stage stage) throws Exception {
        Parent parent = FXMLLoader.load(App.class.getResource("/app.fxml"));
        stage.setTitle("Lab 4. Animals Dataset");
        stage.setScene(new Scene(parent));
        stage.setMinWidth(545);
        stage.setMinHeight(660);
        stage.setMaxHeight(660);
        stage.setMaxWidth(545);
        stage.show();
    }

    public static void main(String[] args) throws Exception {
        launch(args);
    }
}
