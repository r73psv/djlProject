package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class LeNet {
    /**
     * В ходе нашей первой работы с графическими данными мы применили многослойный персептрон к изображениям одежды
     * в наборе данных Fashion-MNIST. Чтобы сделать эти данные доступными для многослойных персептронов,
     * мы сначала выровняли каждое изображение с экрана размером 28×28 матрица преобразуется в файл фиксированной длины 784
     * -трехмерный вектор, а затем обработал их с помощью полностью связанных слоев.
     * Теперь, когда у нас есть управление сверточными слоями, мы можем сохранить пространственную структуру
     * в наших изображениях. В качестве дополнительного преимущества замены плотных слоев сверточными слоями мы
     * получим более экономичные модели (требующие гораздо меньшего количества параметров).
     * В этом разделе мы представим LeNet, одну из первых опубликованных сверточных нейронных сетей, которая привлекла
     * широкое внимание благодаря своей эффективности в задачах компьютерного зрения. Модель была представлена
     * (и названа в честь) Яна Лекуна, в то время исследователя в AT&T Bell Labs, с целью распознавания
     * рукописных цифр на изображениях, опубликованных на LeNet5. Эта работа стала кульминацией десятилетних
     * исследований в области разработки технологии. В 1989 году компания LeCun опубликовала первое исследование,
     * посвященное успешному обучению сверточных нейронных сетей с помощью обратного распространения.
     *
     * В то время LeNet добилась выдающихся результатов, сравнимых с использованием метода опорных векторов (SVM),
     * который в то время был доминирующим подходом к обучению под руководством преподавателя. В конечном итоге
     * LeNet был адаптирован для распознавания цифр при обработке депозитов в банкоматах. По сей день некоторые
     * банкоматы все еще используют код, который Янн и его коллега Леон Ботту написали в 1990-х годах!
     * На высоком уровне LeNet состоит из трех частей: (i) сверточного кодировщика, состоящего из двух сверточных слоев;
     * и (ii) плотного блока, состоящего из трех полностью соединенных слоев;
     * Основными элементами в каждом сверточном блоке являются сверточный слой, функция активации сигмовидной кишки
     * и последующая операция объединения средних значений в пул. Обратите внимание, что, хотя ReLUs и max-пулы работают
     * лучше, эти открытия еще не были сделаны
     * Каждый сверточный слой использует формат 5×5 и функция активации сигмовидной формы. Эти слои отображают
     * пространственно расположенные входные данные на ряд 2D-карт объектов, что обычно увеличивает количество каналов.
     * Первый сверточный слой имеет 6 выходных каналов, в то время как второй - 16. Каждый 2×2 операция объединения
     * в пул (шаг 2) уменьшает размерность в 4 раза за счет уменьшения пространственной дискретизации.
     * Сверточный блок выдает выходные данные с размером, заданным параметром (размер пакета, канал, высота, ширина).
     * Чтобы передать выходные данные из сверточного блока в полностью подключенный блок, мы должны сгладить
     * каждый пример в мини-пакете. Другими словами, мы берем эти 4D-данные и преобразуем их в 2D-данные, ожидаемые
     * от полностью подключенных слоев: напомним, что желаемое нами 2D-представление использует первое измерение
     * для индексации примеров в мини-пакете, а второе - для получения плоского векторного представления каждого примера.
     * Блок полностью подключенных слоев LeNet имеет три полностью подключенных слоя со 120, 84 и 10 выходами
     * соответственно. Поскольку мы все еще проводим классификацию, 10-мерный выходной слой соответствует
     * количеству возможных выходных классов.
     * Хотя для того, чтобы вы по-настоящему поняли, что происходит внутри LeNet, возможно, потребуется немного поработать,
     * мы надеемся, что следующий фрагмент кода убедит вас в том, что реализация таких моделей с помощью современных
     * библиотек глубокого обучения удивительно проста. Нам нужно только создать экземпляр последовательного блока
     * и связать вместе соответствующие слои
     */


    static NDManager manager = NDManager.newBaseManager();
    public static SequentialBlock blocLeNet(){
    Engine.getInstance().setRandomSeed(1111);
    SequentialBlock block = new SequentialBlock();

block
        .add(Conv2d.builder()
        .setKernelShape(new Shape(5, 5))
            .optPadding(new Shape(2, 2))
            .optBias(false)
                .setFilters(6)
                .build())
            .add(Activation::sigmoid)
    .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
            .add(Conv2d.builder()
                .setKernelShape(new Shape(5, 5))
            .setFilters(16).build())
            .add(Activation::sigmoid)
    .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
        // Blocks.batchFlattenBlock()  преобразует входные данные фигуры (batch size, channel, height, width)
        // во входные данные фигуры (batch size,channel * height * width)
            .add(Blocks.batchFlattenBlock())
            .add(Linear
                         .builder()
                .setUnits(120)
                .build())
            .add(Activation::sigmoid)
    .add(Linear
                 .builder()
                .setUnits(84)
                .build())
            .add(Activation::sigmoid)
    .add(Linear
                 .builder()
                .setUnits(10)
                .build());
return block;
    }
/**
 * Мы позволили себе небольшую вольность с исходной моделью, удалив гауссову активацию в последнем слое.
 * В остальном эта сеть соответствует оригинальной архитектуре LeNet5. Мы также создаем модель и объект Trainer,
 * чтобы инициализировать структуру один раз. Путем передачи одноканального (черно-белого) изображения размером 28×28
  * пропустив изображение через сетку и распечатав выходную форму на каждом слое, мы можем проверить модель,
 * чтобы убедиться, что ее работа соответствует тому, что мы ожидаем
 */
static float lr = 0.9f;
public static Model modelLenet(){

    Model model = Model.newInstance("cnn");
    SequentialBlock block=blocLeNet();
    model.setBlock(block);
    return model;
}
static SequentialBlock block =blocLeNet();
static Model model=modelLenet();
    static Loss loss = Loss.softmaxCrossEntropyLoss();

   static Tracker lrt = Tracker.fixed(lr);
    static Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

    static  DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
            .optDevices(Engine.getInstance().getDevices(1)) // Single GPU
            .addEvaluator(new Accuracy()) // Model Accuracy
            .addTrainingListeners(TrainingListener.Defaults.basic());

   static NDArray X = manager.randomUniform(0f, 1.0f, new Shape(1, 1, 28, 28));
    public static Trainer train(){
    Trainer trainer = model.newTrainer(config);
      trainer.initialize(X.getShape());

    Shape currentShape = X.getShape();

    for (int i = 0; i < block.getChildren().size(); i++) {
        Shape[] newShape = block.getChildren().get(i).getValue().getOutputShapes(new Shape[]{currentShape});
        currentShape = newShape[0];
        System.out.println(block.getChildren().get(i).getKey() + " layer output : " + currentShape);
    }
    return trainer;
}
/**
 * Обратите внимание, что высота и ширина представления на каждом слое в сверточном блоке уменьшены
 * (по сравнению с предыдущим слоем). Первый сверточный слой использует 2 * 2 пикселя отступа, чтобы компенсировать
 * уменьшение высоты и ширины, которое в противном случае могло бы возникнуть в результате использования формата 5×5
 * ядер. Напротив, второй сверточный слой исключает заполнение, и, таким образом, высота и ширина уменьшаются в 4 раза.
 * пикселей. По мере того, как мы поднимаемся по стеку слоев, количество каналов увеличивается от слоя к слою
 * с 1 во входных данных до 6 после первого сверточного слоя и 16 после второго слоя. Однако каждый объединяющий
 * слой уменьшает высоту и ширину вдвое. Наконец, каждый полностью подключенный слой уменьшает размерность,
 * в итоге выдавая выходные данные, размерность которых соответствует количеству классов.
 */
static double[] trainLoss;
    static double[] testAccuracy;
    double[] epochCount;
    static double[] trainAccuracy;
public void trainLeNet(){
    int batchSize = 256;
    int numEpochs = Integer.getInteger("MAX_EPOCH", 10);


    epochCount = new double[numEpochs];

    for (int i = 0; i < epochCount.length; i++) {
        epochCount[i] = (i + 1);
    }

    FashionMnist trainIter = FashionMnist.builder()
            .optUsage(Dataset.Usage.TRAIN)
            .setSampling(batchSize, true)
            .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
            .build();


    FashionMnist testIter = FashionMnist.builder()
            .optUsage(Dataset.Usage.TEST)
            .setSampling(batchSize, true)
            .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
            .build();

}

    public static void trainingChapter6(ArrayDataset trainIter, ArrayDataset testIter,
                                        int numEpochs, Trainer trainer) throws IOException, TranslateException {

        double avgTrainTimePerEpoch = 0;
        Map<String, double[]> evaluatorMetrics = new HashMap<>();

        trainer.setMetrics(new Metrics());

        EasyTrain.fit(trainer, numEpochs, trainIter, testIter);

        Metrics metrics = trainer.getMetrics();

        trainer.getEvaluators().stream()
                .forEach(evaluator -> {
                    evaluatorMetrics.put("train_epoch_" + evaluator.getName(), metrics.getMetric("train_epoch_" + evaluator.getName()).stream()
                            .mapToDouble(x -> x.getValue().doubleValue()).toArray());
                    evaluatorMetrics.put("validate_epoch_" + evaluator.getName(), metrics.getMetric("validate_epoch_" + evaluator.getName()).stream()
                            .mapToDouble(x -> x.getValue().doubleValue()).toArray());
                });

        avgTrainTimePerEpoch = metrics.mean("epoch");

        trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
        trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
        testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

        System.out.printf("loss %.3f," , trainLoss[numEpochs-1]);
        System.out.printf(" train acc %.3f," , trainAccuracy[numEpochs-1]);
        System.out.printf(" test acc %.3f\n" , testAccuracy[numEpochs-1]);
        System.out.printf("%.1f examples/sec \n", trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10, 9)));
    }
}