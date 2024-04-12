package com.github.r73pls.djl_Project.mlp;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.Batch;
import ai.djl.training.loss.Loss;
import com.github.r73pls.djl_Project.imageClassificftion.FashionMinst;
import com.github.r73pls.djl_Project.imageClassificftion.TrainingModelSm;
import com.github.r73pls.djl_Project.imageClassificftion.Utils;

import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;
import static com.github.r73pls.djl_Project.mlp.MlpOne.*;

public class MlpTrainer {
    private static int numEpochs=Integer.getInteger("MAX-Epoch", 10);
    private static float lr =0.5f;
    private  static double[] trainLoss = new double[numEpochs];
    private  static double[] testAccuracy = new double[numEpochs];
    private  static double[] epochCount = new double[numEpochs];
    private  static double[] trainAccuracy =  new double[numEpochs];
    private static float epochLoss =0f;
    private static float accuracyVal=0f;
    //функция потерь
    Loss loss=Loss.softmaxCrossEntropyLoss();

    @FunctionalInterface
    public static interface ParamConsumer {
        void accept(NDList params, float lr, int batchSize);
    }
    public static void train(UnaryOperator<NDArray> net, Iterable<Batch> trainIter,Iterable<Batch> testIter, Loss loss, TrainingModelSm.ParamConsumer updater){
        for (int epoch=1; epoch<=numEpochs; epoch++){
            System.out.println("Running epoch " + epoch +"......");
            for (Batch batch: trainIter){
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
                TrainingModelSm.sgd(params, lr, batchSize); // updater
                }
            trainLoss[epoch-1] = epochLoss/trainIter.size();
            trainAccuracy[epoch-1] = accuracyVal/trainIter.size();
разделить?
            epochLoss = 0f;
            accuracyVal = 0f;
            // testing now
            for (Batch batch : testIter) {

                NDArray X = batch.getData().head();
                NDArray y = batch.getLabels().head();

                NDArray yHat = net(X); // net function call
                accuracyVal += Utils.accuracy(yHat, y);
            }

            testAccuracy[epoch-1] = accuracyVal/testIter.size();
            epochCount[epoch-1] = epoch;
            accuracyVal = 0f;
            System.out.println("Finished epoch " + epoch);

             }
        System.out.println("Finished training!");
      }

}


