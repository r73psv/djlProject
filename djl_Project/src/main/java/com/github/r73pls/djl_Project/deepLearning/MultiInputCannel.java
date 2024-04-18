package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class MultiInputCannel {
    /**
     * Когда входные данные содержат несколько каналов, нам нужно создать ядро свертки с таким же количеством входных
     * каналов, что и входные данные, чтобы оно могло выполнять взаимную корреляцию с входными данными.
     * Поскольку входное и свертывающее ядра имеют ci каналов, мы можем выполнить операцию взаимной корреляции
     * с двумерным массивом входных данных и двумерным массивом ядра свертки для каждого канала, добавив ci результаты
     * объединяются (суммируются по каналам) для получения двумерного массива.
     * Чтобы убедиться, что мы действительно понимаем, что здесь происходит, мы можем сами реализовать
     * операции взаимной корреляции с несколькими входными каналами. Обратите внимание, что все, что мы делаем, - это
     * выполняем одну операцию взаимной корреляции для каждого канала, а затем суммируем результаты с помощью функции sum().
     */

    NDManager manager = NDManager.newBaseManager();

    public NDArray corr2D(NDArray X, NDArray K) {

        long h = K.getShape().get(0); //высота ядра свертки
        long w = K.getShape().get(1);//ширина ядра свертка

        //создание целевого массива, заполненного нулевыми значениями
        NDArray Y = manager.zeros(new Shape(X.getShape().get(0) - h + 1, X.getShape().get(1) - w + 1));
        //заполнение целевог массива вычисляемыми значениями
        for (int i = 0; i < Y.getShape().get(0); i++) {
            for (int j = 0; j < Y.getShape().get(1); j++) {
                NDArray temp = X.get(i + ":" + (i + h) + "," + j + ":" + (j + w)).mul(K);
                Y.set(new NDIndex(i + "," + j), temp.sum());
            }
        }
        return Y;
    }

    public static NDArray corr2dMultiIn(NDArray X, NDArray K) {

        long h = K.getShape().get(0);
        long w = K.getShape().get(1);

        // Сначала пройдите по 0-му измерению (размер канала) `X` и `K`. Затем сложите их вместе

        NDArray res = manager.zeros(new Shape(X.getShape().get(0) - h + 1, X.getShape().get(1) - w + 1));
        for (int i = 0; i < X.getShape().get(0); i++) {
            for (int j = 0; j < K.getShape().get(0); j++) {
                if (i == j) {
                    res = res.add(corr2D(X.get(new NDIndex(i)), K.get(new NDIndex(j))));
                }
            }
        }
        return res;
    }

     // Мы можем построить входной массив X и массив ядра K для проверки результатов операции взаимной корреляции.

    public void test(){
        NDArray X = manager.create(new Shape(2, 3, 3), DataType.INT32);
        X.set(new NDIndex(0), manager.arange(9));
        X.set(new NDIndex(1), manager.arange(1, 10));
        X = X.toType(DataType.FLOAT32, true);

        NDArray K = manager.create(new Shape(2, 2, 2), DataType.INT32);
        K.set(new NDIndex(0), manager.arange(4));
        K.set(new NDIndex(1), manager.arange(1, 5));
        K = K.toType(DataType.FLOAT32, true);

        corr2dMultiIn(X, K);
    }
}
