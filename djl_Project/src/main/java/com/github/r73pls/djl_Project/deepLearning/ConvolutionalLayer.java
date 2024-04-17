package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

public class ConvolutionalLayer {

    /**
     * Сверточный слой выполняет взаимную корреляцию входных данных и ядер и добавляет скалярное смещение для получения
     * выходных данных. Двумя параметрами сверточного слоя являются ядро и скалярное смещение. При обучении моделей,
     * основанных на сверточных слоях, мы обычно инициализируем ядра случайным образом, точно так же, как мы бы это
     * сделали с полностью подключенным слоем.
     * Теперь мы готовы к реализации двумерного сверточного слоя на основе функции corr 2d.
     * В функции конструктора сверточного слоя мы объявляем weight и bias в качестве двух параметров класса.
     * Функция прямого вычисления forward вызывает функцию corr2d и добавляет смещение. Как и в случае с h×w
     * взаимная корреляция мы также называем сверточные слои h×w
     */
    private NDArray w;
    private NDArray b;

    public NDArray getW(){
        return w;
    }

    public NDArray getB(){
        return b;
    }

    public ConvolutionalLayer(Shape shape){
        NDManager manager = NDManager.newBaseManager();
        w = manager.create(shape);
        b = manager.randomNormal(new Shape(1));
        w.setRequiresGradient(true);
    }

    public NDArray forward(NDArray X){
        return Corr2D.corr2d(X, w).add(b);
    }
}
