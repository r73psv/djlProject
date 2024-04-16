package com.github.r73pls.djl_Project.mlp;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import com.github.r73pls.djl_Project.imageClassificftion.Utils;

import java.io.IOException;
import java.util.function.UnaryOperator;

import static com.github.r73pls.djl_Project.imageClassificftion.DataSetImage.batchSize;
import static com.github.r73pls.djl_Project.mlp.MlpOne.*;

public class MlpTrainer {
    private static final int numEpochs=Integer.getInteger("MAX-Epoch", 10);
    private static final float lr =0.5f;
    private  static final double[] trainLoss = new double[numEpochs];
    private  static final double[] testAccuracy = new double[numEpochs];
    private  static final double[] epochCount = new double[numEpochs];
    private  static final double[] trainAccuracy =  new double[numEpochs];
    private static float epochLoss =0f;
    private static float accuracyVal=0f;
    //функция потерь
    static Loss loss=Loss.softmaxCrossEntropyLoss();

    @FunctionalInterface
    public static interface ParamConsumer {
        void accept(NDList params, float lr, int batchSize);
    }
    public static void train(UnaryOperator<NDArray> net, Dataset trainDataset, Dataset testDataset, ParamConsumer updater) throws TranslateException, IOException {
        for (int epoch=1; epoch<=numEpochs; epoch++){
            System.out.println("Running epoch " + epoch +"......");

            for (Batch batch: trainDataset.getData(manager)){
                NDArray X  = batch.getData().head();
                NDArray y = batch.getLabels().head();
                try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                    NDArray yHat = net(X);
                    NDArray lossValue = loss.evaluate(new NDList(y), new NDList(yHat));
                    NDArray l = lossValue.mul(batchSize);

                    accuracyVal += Utils.accuracy(yHat, y);
                    epochLoss += l.sum().getFloat();

                    gc.backward(l); // gradient calculation
                }
                batch.close();

                //TrainingModelSm.sgd(params, lr, batchSize); // updater

                updater.accept(params, lr, batch.getSize());  // Update parameters using their gradient
            }

//            trainLoss[epoch-1] = epochLoss/trainDataset.size();
//            trainAccuracy[epoch-1] = accuracyVal/trainDataset.size();

            epochLoss = 0f;
            accuracyVal = 0f;
            // testing now

            for (Batch batch : testDataset.getData(manager)) {

                NDArray X = batch.getData().head();
                NDArray y = batch.getLabels().head();

                NDArray yHat = net(X); // net function call
                accuracyVal += Utils.accuracy(yHat, y);
            }


//            testAccuracy[epoch-1] = accuracyVal/testIter.size();
            epochCount[epoch-1] = epoch;
            accuracyVal = 0f;
            System.out.println("Finished epoch " + epoch);

        }
        System.out.println("Finished training!");
    }
}


