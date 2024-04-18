package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;

public class MultiOutCannel {
    /**
     *  В самых популярных архитектурах нейронных сетей мы фактически увеличиваем размер канала по мере продвижения
     *  вверх по нейронной сети, обычно уменьшая дискретизацию, чтобы уменьшить пространственное разрешение и увеличить
     *  глубину канала. Интуитивно вы можете представить, что каждый канал отвечает за определенный набор функций.
     *  Реальность немного сложнее, чем самые наивные интерпретации этой интуиции, поскольку представления не изучаются
     *  независимо, а скорее оптимизируются для совместной работы. Таким образом, возможно, что не один канал изучает
     *  детектор границ, а какое-то направление в пространстве канала соответствует обнаружению границ.
     */
    NDManager manager=NDManager.newBaseManager();


    public NDArray corrMultiInOut(NDArray X, NDArray K) {

        long cin = K.getShape().get(0); // номер ядра (?)
        long h = K.getShape().get(2);   //высота ядра
        long w = K.getShape().get(3);   //ширина ядра

        // Пройдите по 0-му измерению `K` и каждый раз выполняйте операции взаимной корреляции с входными данными `X`.
        // Все результаты объединяются вместе с помощью функции stack

        NDArray res = manager.create(new Shape(cin, X.getShape().get(1) - h + 1, X.getShape().get(2) - w + 1));

        for (int j = 0; j < K.getShape().get(0); j++) {
            res.set(new NDIndex(j), MultiInputCannel.corr2dMultiIn(X, K.get(new NDIndex(j))));
        }

        return res;
    }
    //создаем ядро свертки с 3 выходными каналами,
    // объединяя массив ядра K с K + 1 (плюс по одному для каждого элемента в K) и K + 2.
    // K = NDArrays.stack(new NDList(K, K.add(1), K.add(2)));
    // K.getShape();

    /*
    Ниже мы выполняем операции взаимной корреляции входного массива X с массивом ядра K.
    Теперь выходные данные содержат 3 канала. Результат первого канала согласуется с результатом предыдущего
    входного массива X и ядра с несколькими входными каналами и одним выходным каналом.
    corrMultiInOut(X, K);
     */

    /**
     * На первый взгляд, свертка 1×1, т.е. kh=kw =1, тем не менее, это популярные операции, которые иногда
     * включаются в проекты сложных глубоких сетей. Давайте рассмотрим более подробно, что это на самом деле делает.
     *Поскольку используется минимальное окно, свертка размером 1×1 не позволяет более крупным сверточным слоям
     * распознавать шаблоны, состоящие из взаимодействий между соседними элементами в измерениях высоты и ширины.
     * Единственное вычисление в формате 1×1 выполняется в размерности канала.
     *Давайте проверим, работает ли это на практике: мы реализуем схему 1×1  с использованием полностью подключенного слоя.
     * Единственное, что нам нужно сделать, это внести некоторые коррективы в форму данных до и после матричного умножения.
     */

public NDArray corr2dMultiInOut1x1(NDArray X, NDArray K) {

    long channelIn = X.getShape().get(0);
    long height = X.getShape().get(1);
    long width = X.getShape().get(2);

    long channelOut = K.getShape().get(0);
    X = X.reshape(channelIn, height * width);
    K = K.reshape(channelOut, channelIn);
    NDArray Y = K.dot(X); // Matrix multiplication in the fully connected layer

    return Y.reshape(channelOut, height, width);
    }

    /*
    При выполнении 1×1 приведенная выше функция эквивалентна ранее реализованной функции
    взаимной корреляции corrMultiInOut(). Давайте проверим это с помощью некоторых справочных данных.

    X = manager.randomUniform(0f, 1.0f, new Shape(3, 3, 3));
    K = manager.randomUniform(0f, 1.0f, new Shape(2, 3, 1, 1));
    NDArray Y1 = corr2dMultiInOut1x1(X, K);
    NDArray Y2 = corrMultiInOut(X, K);
    System.out.println(Math.abs(Y1.sum().getFloat() - Y2.sum().getFloat()) < 1e-6);
     */


}
