package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;

public class Corr2D {
    /**
     * В операции двумерной взаимной корреляции мы начинаем с окна свертки, расположенного в верхнем левом углу
     * входного массива, и перемещаем его по входному массиву как слева направо, так и сверху вниз.
     * Когда окно свертки перемещается в определенное положение, входной подмассив, содержащийся в этом окне,
     * и массив ядра перемножаются (поэлементно), и результирующий массив суммируется, получая одно скалярное значение.
     * Этот результат дает значение выходного массива в соответствующем местоположении.
     * Здесь выходной массив имеет высоту 2 и ширину 2, а четыре элемента получены в результате операции
     * двумерной взаимной корреляции.
     * По каждой оси выходные данные немного меньше входных. Поскольку ширина и высота ядра больше единицы,
     * мы можем правильно вычислить взаимную корреляцию только для тех мест, где ядро полностью вписывается
     * в изображение, размер выходных данных определяется размером входных данных H×W минус размер сверточного ядра h×w
     * через (H−h+1)×(W−w+1). Это так, поскольку нам нужно достаточно места, чтобы "сдвинуть" сверточное ядро по всему
     * изображению (позже мы увидим, как сохранить размер неизменным, заполнив изображение нулями по краям таким образом,
     * чтобы было достаточно места для смещения ядра). Далее мы реализуем этот процесс в функции corr2d,
     * которая принимает входной массив X и массив ядра K и возвращает выходной массив Y.
     */
    static NDManager manager=NDManager.newBaseManager();
public static NDArray corr2d(NDArray X, NDArray K){
    // Compute 2D cross-correlation.
    int h = (int) K.getShape().get(0);
    int w = (int) K.getShape().get(1);

    NDArray Y = manager.zeros(new Shape(X.getShape().get(0) - h + 1, X.getShape().get(1) - w + 1));

    for(int i=0; i < Y.getShape().get(0); i++){
        for(int j=0; j < Y.getShape().get(1); j++){
            Y.set(new NDIndex(i + "," + j), X.get(i + ":" + (i+h) + "," + j + ":" + (j+w)).mul(K).sum());
        }
    }

    return Y;
}

}
