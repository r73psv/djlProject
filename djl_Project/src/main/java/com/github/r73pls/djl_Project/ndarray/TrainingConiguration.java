package com.github.r73pls.djl_Project.ndarray;

import ai.djl.ndarray.NDManager;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

public class TrainingConiguration {
  protected static NDManager manager=NDManager.newBaseManager();

    /**
     * В DJL класс Loss определяет различные функции потерь. Мы будем использовать импортированный класс Loss.
     * В этом примере мы будем использовать реализацию LOSS в квадрате (L2Loss) в DJL.
     */
   protected static Loss l2loss=Loss.l2Loss();

    /**
     * Minibatch SGD и связанные с ним варианты являются стандартными инструментами для оптимизации нейронных сетей,
     * и, таким образом, DJL поддерживает SGD наряду с рядом вариаций этого алгоритма через свой класс оптимизаторов.
     * Когда мы создадим оптимизатор, мы укажем алгоритм оптимизации, который мы хотим использовать (sgd).
     * Мы также можем вручную задать гиперпараметры. Для SGD просто требуется скорость обучения, здесь мы устанавливаем
     * ее на фиксированное значение 0,03.
     */

    protected static Tracker ltr= Tracker.fixed(0.03f);
    protected static Optimizer sgd =Optimizer.sgd().setLearningRateTracker(ltr).build();

    /**
     *Теперь мы создадим конфигурацию обучения, описывающую, как мы хотим обучать нашу модель.
     * Затем мы инициализируем инструктора, который проведет обучение за нас.
     */
    public static DefaultTrainingConfig getConfig() {
        DefaultTrainingConfig config = new DefaultTrainingConfig(l2loss)
                .optOptimizer(sgd) // Optimizer (loss function)
                .optDevices(manager.getEngine().getDevices(1)) // single GPU
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging
        return config;
    }
}
