package com.github.r73pls.djl_Project.imageClassificftion;

import ai.djl.ndarray.NDList;

public class Updater {

    public static void updater(NDList params, float lr, int batchSize) {
        TrainingModelSm.sgd(params, lr, batchSize);
    }
}
