package com.github.r73pls.djl_Project.mlp;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.loss.Loss;

public class MlpOne {
    /**
     *представим наши параметры с помощью нескольких массивов данных. Обратите внимание, что для каждого слоя мы должны
     * отслеживать одну матрицу весов и один вектор смещения. Как всегда, мы вызываем функцию attachGradient(),
     * чтобы выделить память для градиентов (потерь) по отношению к этим параметрам.
     */
    private static int numInputs=784;//размер входного слоя
    private static int numHidden=256;//размер скрытого слоя
    private static int numOutputs=10;//размер выходного слоя
    static NDManager manager=NDManager.newBaseManager();
    static NDArray W1 =manager.randomNormal(0,0.01f,new Shape(numInputs,numHidden), DataType.FLOAT32); //веса скрытого слоя
    static NDArray b1=manager.zeros(new Shape(numHidden));//смещение для скрытого слоя
    static NDArray W2 =manager.randomNormal(0,0.01f,new Shape(numHidden,numOutputs), DataType.FLOAT32); //веса выходного слоя
    static NDArray b2=manager.zeros(new Shape(numOutputs));//смещение входного слоя
    static NDList params =new NDList(W1,b1,W2,b2);

    //выделоние памяти для градиентов
    public void  memoryAllocate(){
        for (NDArray param: params){
            param.setRequiresGradient(true);
        }
    }

    /**
     * Чтобы убедиться, что мы знаем, как все работает, мы сами реализуем активацию ReLU, используя функцию maximum,
     * а не вызывая активацию напрямую.relu.
     */

    public static NDArray relu(NDArray X){
        return X.maximum(0f);
    }

    /**
     * Поскольку мы не учитываем пространственную структуру, мы преобразуем каждое 2D-изображение
     * в плоский вектор числовой длины. Наконец, мы реализуем нашу модель всего несколькими строками кода
     */

    public static NDArray net(NDArray X){
        X=X.reshape(new Shape(-1,numInputs));
        NDArray H=relu(X.dot(W1).addi(b1));
        return H.dot(W2).add(b2);
    }




}
