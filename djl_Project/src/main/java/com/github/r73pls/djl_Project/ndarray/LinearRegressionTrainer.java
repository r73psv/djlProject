package com.github.r73pls.djl_Project.ndarray;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.ParameterList;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class LinearRegressionTrainer {

    private static Model model=RegressionModel.getModel();
    private static DefaultTrainingConfig config =TrainingConiguration.getConfig();
    private static Trainer trainer=model.newTrainer(config);
    private static  ArrayDataset dataset =DataGenerator.DataSetMaker(new float []{2,-3.4f},4.2f,10000,10);
       public static void trainModel(int batchSize,int numEpochs){
        /**
         * Перед обучением нашей модели нам необходимо инициализировать параметры модели, такие как веса и смещения
         * в модели линейной регрессии. Мы просто вызываем функцию initialize с формой модели, которую мы обучаем.
         */
        // First axis is batch size - won't impact parameter initialization
        // Second axis is the input size
        trainer.initialize(new Shape(batchSize, 2));
        /**
         * Обычно DJL не записывает метрики, если это не указано явно, поскольку запись метрик влияет на оптимизацию
         * процесса выполнения. Чтобы записать метрики, мы должны создать экземпляр metrics извне объекта trainer и
         * затем передать его в систему.
         */
        Metrics metrics = new Metrics();
        trainer.setMetrics(metrics);
        /**
         * в течение некоторого количества эпох мы будем выполнять полный проход по набору данных (train_data), итеративно
         * захватывая один мини-набор входных данных и соответствующие метки базовой истинности. Для каждого мини-набора
         * мы выполняем следующее:
         * Генерируйте прогнозы, рассчитывайте потери и градиенты, вызывая trainBatch (пакетный) (прямой и обратный проходы).
         * Обновляйте параметры модели, вызывая функцию step.
         * При ведении журнала автоматически выводятся оценщики, за которыми мы наблюдаем в течение каждой эпохи
         */
        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            System.out.printf("Epoch %d\n", epoch);
            // Iterate over dataset
            try {
                for (Batch batch : trainer.iterateDataset(dataset)) {
                    // Update loss and evaulator
                    EasyTrain.trainBatch(trainer, batch);

                    // Update parameters
                    trainer.step();

                    batch.close();
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }
            // reset training and validation evaluators at end of epoch
            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
        }

    }


    public static void testModel(){
        Block lauer= model.getBlock();
        ParameterList params =lauer.getParameters();
        NDArray wParams = params.valueAt(0).getArray();
        NDArray bParams = params.valueAt(1).getArray();
        float[] w = new float []{2,-3.4f};
        System.out.printf("Error in estimating w: [%f %f]\n", w[0], w[1]);
        System.out.printf("Error in estimating b: %f\n", 4.2f - bParams.getFloat());
    }

public static void saveModel(int numEpochs){
    Path modelDir = Paths.get("D:\\javaProjcts\\djlProject\\models\\");
    try {
        Files.createDirectories(modelDir);
    } catch (IOException e) {
        throw new RuntimeException(e);
    }
    model.setProperty("Epoch", Integer.toString(numEpochs)); // save epochs trained as metadata

    try {
        model.save(modelDir, "lin-reg");
    } catch (IOException e) {
        throw new RuntimeException(e);
    }

}

}
