package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Utils;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class LoadSaveTensor {
    /**
     * Для отдельных тензоров мы можем преобразовать NDArrays в byte[]s, вызвав их функцию encode(). Затем мы можем
     * преобразовать их обратно в NDArrays, вызвав функцию NDArray decode() и передав NDManager(для управления
     * созданным NDArray) и byte[] (требуемый тензор).
     * Затем мы можем использовать FileInputStream и FileOutputStream для чтения и записи этих данных в файлы соответственно.
     */
    public LoadSaveTensor() throws FileNotFoundException, IOException {
    }
    NDManager manager = NDManager.newBaseManager();
    NDArray x = manager.arange(4);
    NDArray x2 = manager.arange(4);
    //запись в файл
    public void  saveTensor(NDArray x, String fileName) throws IOException {

        try (FileOutputStream fos = new FileOutputStream(fileName)) {
            fos.write(x.encode());
        }

    }

    //чтение из файла

    public NDArray loadTensor(String fileName) throws IOException {
        NDArray x;
        try (FileInputStream fis = new FileInputStream(fileName)) {
            // Мы используем метод `Utils``toByteArray()` для чтения из `FileInputStream` и возвращаем его в виде `byte[]`
            x = NDArray.decode(manager, Utils.toByteArray(fis));
        }
        return x;
    }

//Мы также можем сохранить NDList в файл и загрузить его обратно:
NDList list = new NDList(x, x2);
    public void saveNDlist(NDList list, String fileName) throws IOException {
            try (FileOutputStream fos = new FileOutputStream(fileName)) {
            fos.write(list.encode());
        }
    }
    public NDList loadNDlist(String fileName) throws IOException {
        NDList ndList;
        try (FileInputStream fis = new FileInputStream(fileName)) {
            ndList = NDList.decode(manager, Utils.toByteArray(fis));
        }
        return ndList;
    }
}
