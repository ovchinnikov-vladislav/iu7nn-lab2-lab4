package nn.ovchinnikov.lab2.classification.util;

import java.io.File;
import java.util.Objects;

public class UtilFile {

    public static boolean recursiveDeleteModel(String path) {
        File file = new File(path);
        if (!file.exists())
            return true;
        if (file.isDirectory()) {
            for (File f : Objects.requireNonNull(file.listFiles())) {
                recursiveDeleteModel(f.getPath());
            }
        }
        return file.delete();
    }

}
