package com.github.r73pls.djl_Project.imageClassificftion;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

import static com.github.r73pls.djl_Project.imageClassificftion.Net.*;
import static com.github.r73pls.djl_Project.imageClassificftion.Utils.*;

public class TrainingModelSm {
    @FunctionalInterface
    public static interface ParamConsumer {
        void accept(NDList params, float lr, int batchSize);
    }
    public static void sgd(NDList params, float lr, int batchSize) {
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            // Update param
            // param = param - param.gradient * lr / batchSize
            param.subi(param.getGradient().mul(lr).div(batchSize));
        }
    }
    static NDManager manager = NDManager.newBaseManager();
    public static float[] trainEpochCh3(UnaryOperator<NDArray> net, Iterable<Batch> trainIter, BinaryOperator<NDArray> loss, ParamConsumer updater) {
        Accumulator metric = new Accumulator(3); // trainLossSum, trainAccSum, numExamples

        // Attach Gradients
        for (NDArray param : params) {
            param.setRequiresGradient(true);
        }

        for (Batch batch : trainIter) {
            NDArray X = batch.getData().head();
            NDArray y = batch.getLabels().head();
            X = X.reshape(new Shape(-1, 784));

            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                // Minibatch loss in X and y
                NDArray yHat = net.apply(X);
                NDArray l = loss.apply(yHat, y);
                gc.backward(l);  // Compute gradient on l with respect to w and b
                metric.add(new float[]{l.sum().toType(DataType.FLOAT32, false).getFloat(),
                        accuracy(yHat, y),
                        (float)y.size()});
                gc.close();
            }
            updater.accept(params, 0.1f, batch.getSize());  // Update parameters using their gradient

            batch.close();
        }
        // Return trainLoss, trainAccuracy
        return new float[]{metric.get(0) / metric.get(2), metric.get(1) / metric.get(2)};
    }

    public static void trainCh3(UnaryOperator<NDArray> net, Dataset trainDataset, Dataset testDataset,
                         BinaryOperator<NDArray> loss, int numEpochs, ParamConsumer updater)
            throws IOException, TranslateException {
        Animator animator = new Animator();
        for (int i = 1; i <= numEpochs; i++) {
            float[] trainMetrics = trainEpochCh3(net, trainDataset.getData(manager), loss, updater);
            float accuracy = Utils.evaluateAccuracy(net, testDataset.getData(manager));
            float trainAccuracy = trainMetrics[1];
            float trainLoss = trainMetrics[0];

            animator.add(i, accuracy, trainAccuracy, trainLoss);
            System.out.printf("Epoch %d: Test Accuracy: %f\n", i, accuracy);
            System.out.printf("Train Accuracy: %f\n", trainAccuracy);
            System.out.printf("Train Loss: %f\n", trainLoss);
        }
    }
}
