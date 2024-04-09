package com.github.r73pls.djl_Project.ndarray;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.ArrayDataset;

public class DataGenerator {
    static NDManager ndManager =NDManager.newBaseManager();
    public static DataPoints syntheticData(NDManager ndManager, NDArray w, float b, int numExamples){
        NDArray X =ndManager.randomNormal(new Shape(numExamples, w.size()));
        NDArray y = X.dot(w).add(b);
        //добавление шума
        y=y.add(ndManager.randomNormal(0,0.01f,y.getShape(), DataType.FLOAT32));
        return  new DataPoints(X,y);
    }

    /**
     * Вызваем пакет dataset от DJL для чтения данных. Первым шагом будет создание экземпляра ArrayDataset.
     * Здесь мы задаем объекты и метки в качестве параметров. Мы также указываем размер пакета и логическое
     * значение shuffle, указывающее, хотим ли мы, чтобы ArrayDataset произвольно отбирал данные.
     */
    public static ArrayDataset loadArray(NDArray features, NDArray labels, int bachSize, boolean shuffle){
        return new  ArrayDataset.Builder()
                .setData(features) //загружаем данные
                .optLabels(labels) //загружаем метки
                .setSampling(bachSize,shuffle) //устанавливаем размер пакета
                .build();
    }
    public static ArrayDataset DataSetMaker (float[] w, float b, int numExamples, int bachSize){
        //веса
        NDArray trueW = ndManager.create(w);
        //b- смещение
        //numExamples -количество
        //bachSize - размер пакета
        DataPoints dp = syntheticData(ndManager,trueW,b,numExamples);
        NDArray features = dp.getX();
        NDArray labels = dp.getY();
        ArrayDataset dataset= loadArray(features,labels,bachSize,false);
        return  dataset;

    }
}
