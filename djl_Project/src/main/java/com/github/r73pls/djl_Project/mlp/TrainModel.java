package com.github.r73pls.djl_Project.mlp;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.metric.*;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class TrainModel {
    protected static int numEpoch=Integer.getInteger("MAX_EPOCH",10);
    static Model model= modelMlp.model();
    public static void train(Dataset trainIter, Dataset testIter){
        int bathSize=256;

        double[]trainLoss;
        trainLoss =new double[numEpoch];
        double[]testAccuracy;
        testAccuracy =new double[numEpoch];
        double[]epocCount;
        epocCount =new double[numEpoch];
        double[]trainAccuracy;
        trainAccuracy =new double[numEpoch];


        for (int i=0;i<epocCount.length;i++){
            epocCount[i]=i+1;
        }

        Map<String,double[]> evaluatorMetrics=new HashMap<>();
        Tracker lrt =Tracker.fixed(0.5f);
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();
        Loss loss =Loss.softmaxCrossEntropyLoss();

        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optOptimizer(sgd)
                .optDevices(Engine.getInstance().getDevices(1))
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());

        try(Trainer trainer=model.newTrainer(config)){
            trainer.initialize(new Shape(1,784));
            trainer.setMetrics(new Metrics());


            try {
                EasyTrain.fit(trainer,numEpoch,trainIter, testIter);
            } catch (IOException e) {
                throw new RuntimeException(e);
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }

            Metrics metrics = trainer.getMetrics();
            trainer.getEvaluators().stream()
                    .forEach(evaluator -> {
                        evaluatorMetrics.put("train_epoch_"+evaluator.getName(),metrics.getMetric("train_epoch_"+evaluator.getName()).stream()
                                .mapToDouble(x-> x.getValue().doubleValue()).toArray());
                        evaluatorMetrics.put("validate_epoch_"+evaluator.getName(),metrics.getMetric("validate_epoch_"+evaluator.getName()).stream()
                                .mapToDouble(x-> x.getValue().doubleValue()).toArray());
        });
     }

   }
    public static void saveModel(){
        Path modelDir = Paths.get("D:\\javaProjcts\\djlProject\\models\\");
        try {
            Files.createDirectories(modelDir);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        model.setProperty("Epoch", Integer.toString(numEpoch)); // save epochs trained as metadata

        try {
            model.save(modelDir, "mlp");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }
}

