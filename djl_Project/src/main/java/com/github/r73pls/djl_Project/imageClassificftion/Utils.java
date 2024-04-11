package com.github.r73pls.djl_Project.imageClassificftion;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.TranslateException;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.function.UnaryOperator;

public class Utils {

    public static float accuracy (NDArray yHat, NDArray y){
        // Проверяем размер 1-го измерения, чтобы увидеть, есть ли у нас несколько образцов
        if (yHat.getShape().size(1) > 1) {
            // Argmax получает индекс максимального числа аргументов для данной оси 1
            // Преобразует yHat в тот же тип данных, что и y (int32)
            // Подсчитывает количество истинных записей
            return yHat.argMax(1).toType(DataType.INT32, false).eq(y.toType(DataType.INT32, false))
                    .sum().toType(DataType.FLOAT32, false).getFloat();
        }
        return yHat.toType(DataType.INT32, false).eq(y.toType(DataType.INT32, false))
                .sum().toType(DataType.FLOAT32, false).getFloat();

    }
    public static float evaluateAccuracy(UnaryOperator<NDArray> net, Iterable <Batch> dataIterator) {
        Accumulator metric = new Accumulator (2);  // numCorrectedExamples, numExamples
        Batch batch = dataIterator.iterator().next();
        NDArray X = batch.getData().head();
        NDArray y = batch.getLabels().head();
        metric.add(new float[]{accuracy(net.apply(X), y), (float)y.size()});
        batch.close();

        return metric.get(0) / metric.get(1);
    }

//    public static BufferedImage predictCh3(UnaryOperator<NDArray> net, ArrayDataset dataset, int number, NDManager manager)
//            throws IOException, TranslateException {
//        int[] predLabels = new int[number];
//
//        Batch batch = dataset.getData(manager).iterator().next();
//        NDArray X = batch.getData().head();
//        int[] yHat = net.apply(X).argMax(1).toType(DataType.INT32, false).toIntArray();
//        for (int i = 0; i < number; i++) {
//            predLabels[i] = yHat[i];
//        }
//
//        return FashionMnistUtils.showImages(dataset, predLabels, 28, 28, 4, manager);
//    }
}
