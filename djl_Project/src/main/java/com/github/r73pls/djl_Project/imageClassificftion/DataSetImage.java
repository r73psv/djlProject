package com.github.r73pls.djl_Project.imageClassificftion;
import ai.djl.basicdataset.cv.classification.*;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;

import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Color;
import java.io.IOException;

public class DataSetImage {
    public static int batchSize = 256;
    private static boolean randomShuffle = true;
    NDManager manager= NDManager.newBaseManager();
    /**
     * сначала определим функцию getDataset(), которая получает и считывает набор данных Fashion-MNIST.
     * Она возвращает набор данных для обучающего набора или набора проверки, в зависимости от переданного в usage
     * атем вы можете вызвать getData(менеджер) для набора данных, чтобы получить соответствующий итератор.
     * Он также принимает значения BatchSize и randomShuffle, которые определяют размер каждого пакета и
     * необходимость случайной перетасовки данных соответственно.
     */

    public static void getDataSet() {
        FashionMnist mnistTrain = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();
        FashionMnist mnistTest = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();
        try {
            mnistTrain.prepare();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        try {
            mnistTest.prepare();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }

}