package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.loss.Loss;

public class ObjectEdgeDetection {
    /**
     * Простое применение сверточного слоя: определение края объекта на изображении путем определения местоположения
     * изменения пикселя. Сначала мы создаем "изображение’ размером 6×8 пикселей. Средние четыре столбца - черные (0),
     * а остальные - белые (1).
     */
    NDManager manager=NDManager.newBaseManager();
    NDArray X = manager.ones(new Shape(6,8));
    NDArray K;
    NDArray Y;
    public void createImage(){

        X.set(new NDIndex(":" + "," + 2 + ":" + 6), 0f);
        System.out.println(X);
        /*
        * Далее мы создаем ядро K высотой 1 и шириной 2. Когда мы выполняем операцию взаимной корреляции
        * с входными данными, если соседние по горизонтали элементы
        * совпадают, выходные данные равны 0. В противном случае выходные данные будут отличны от нуля.
        */

         K = manager.create(new float[]{1, -1}, new Shape(1,2));

        /*
        Мы готовы выполнить операцию взаимной корреляции с аргументами X (наши входные данные) и K (наше ядро).
        Как вы можете видеть, мы определяем 1 для границы от белого к черному и -1 для границы от черного к белому.
        Все остальные выходные данные принимают значение 00.
        */
        Y = Corr2D.corr2d(X, K);
        System.out.println(Y);
        /*
        Теперь мы можем применить ядро к транспонированному изображению. Как и ожидалось, оно исчезает.
        Ядро K распознает только вертикальные края.
         */
        Corr2D.corr2d(X.transpose(), K);

    }

    /**
     * Теперь давайте посмотрим, сможем ли мы узнать ядро, которое сгенерировало Y из X, просмотрев только пары (вход, выход).
     * Сначала мы создаем сверточный слой и инициализируем его ядро как случайный массив. Далее, на каждой итерации,
     * мы будем использовать квадрат ошибки для сравнения Y с результатами сверточного слоя. Затем мы можем рассчитать
     * градиент, чтобы обновить вес. Для простоты в этом сверточном слое мы будем игнорировать смещение.
     * На этот раз мы будем использовать встроенный класс Block и Conv2d из DJL.
     */
public void imageDetection(){

    X = X.reshape(1,1,6,8);
    Y = Y.reshape(1,1,6,7);

    Loss l2Loss = Loss.l2Loss();
    // Построим двумерный сверточный слой с 1 выходным каналом и
    // ядром формы (1, 2). Для простоты мы игнорируем здесь смещение
    Block block = Conv2d.builder()
            .setKernelShape(new Shape(1, 2))
            .optBias(false)
            .setFilters(1)
            .build();

    block.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
    block.initialize(manager, DataType.FLOAT32, X.getShape());

// Двумерный сверточный слой использует четырехмерный ввод и  вывод в формате (example, channel, height, width),
// где batch size(количество примеров в пакете) и количество каналов равны 1

    ParameterList params = block.getParameters();
    NDArray wParam = params.get(0).getValue().getArray();
    wParam.setRequiresGradient(true);

    NDArray lossVal = null;
    ParameterStore parameterStore = new ParameterStore(manager, false);

        for (int i = 0; i < 10; i++) {

        wParam.setRequiresGradient(true);

        try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
            NDArray yHat = block.forward(parameterStore, new NDList(X), true).singletonOrThrow();
            NDArray l = l2Loss.evaluate(new NDList(Y), new NDList(yHat));
            lossVal = l;
            gc.backward(l);
        }
        // Update the kernel
        wParam.subi(wParam.getGradient().mul(0.40f));

        if((i+1)%2 == 0){
            System.out.println("batch " + (i+1) + " loss: " + lossVal.sum().getFloat());
        }
    }
        /*
        Обратите внимание, что после 10 итераций ошибка снизилась до небольшого значения.
        Теперь мы рассмотрим изученный нами массив ядра.
         */
    ParameterList params1 = block.getParameters();
    NDArray wParam1 = params.get(0).getValue().getArray();
    System.out.println(wParam1);
    /*
    Действительно, изученный массив ядра приближается к массиву ядра K, который мы определили ранее.
     */
}
}
