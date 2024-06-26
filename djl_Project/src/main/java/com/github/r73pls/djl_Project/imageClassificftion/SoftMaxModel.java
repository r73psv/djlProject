package com.github.r73pls.djl_Project.imageClassificftion;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

public class SoftMaxModel {
   static NDManager manager =NDManager.newBaseManager();

    /**
     *  Выходной слой регрессии softmax является полностью связанным слоем.
     *  Таким образом, для реализации нашей модели нам просто нужно добавить один связанный слой с 10 выходами
     *  в нашу последовательность. Опять же, здесь последовательность на самом деле не обязательна,
     *  но мы могли бы с таким же успехом сформировать привычку, поскольку она будет повсеместной при внедрении глубоких моделей.
     */
    public static Model model(){
        Model model= Model.newInstance("Softmax-Model");
        SequentialBlock net = new SequentialBlock();
        net.add(Blocks.batchFlattenBlock(28*28));//входной блок 28*28=784
        net.add(Linear.builder().setUnits(10).build());//выходной блок
        model.setBlock(net);
        return model;
       }

    /**
     * Мы хотим сохранить обычную функцию softmax под рукой на тот случай, если нам когда-нибудь
     * захочется оценить вероятности, выводимые нашей моделью. Но вместо того, чтобы передавать вероятности softmax
     * в нашу новую функцию потерь, мы просто передадим логиты и сразу вычислим softmax и его логарифмическую функцию
     * потерь softmaxCrossEntropy, которая выполняет такие умные действия, как трюк log-sum-exp (смотрите в Википедии).
     */
    static Loss loss =Loss.softmaxCrossEntropyLoss();

    /**
     * Здесь мы используем стохастический градиентный спуск с минибатчами со скоростью обучения 0,1
     * в качестве алгоритма оптимизации. Обратите внимание, что это то же самое, что мы применили в примере
     * с линейной регрессией, и это иллюстрирует общую применимость оптимизаторов.
     */

    static Tracker lrt = Tracker.fixed(0.1f);
    static Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

    /**
     * Теперь мы создадим конфигурацию обучения, описывающую, как мы хотим обучать нашу модель.
     * Затем мы создадим инструктора, который проведет обучение за нас.
     */
    static DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
            .optOptimizer(sgd) // Optimizer
            .optDevices(manager.getEngine().getDevices(1)) // single GPU
            .addEvaluator(new Accuracy()) // Model Accuracy
            .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

   public static Trainer trainer = SoftMaxModel.model().newTrainer(config);

}
