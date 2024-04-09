package com.github.r73pls.djl_Project.ndarray;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.loss.L2Loss;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

public class RegressionModel {


/**
 *  Для стандартных операций мы можем использовать предопределенные блоки DJL, которые позволяют нам сосредоточиться
 *  на слоях, используемых для построения модели, а не на реализации. Чтобы определить линейную модель, мы сначала
 *  импортируем класс Model, который определяет множество полезных методов для взаимодействия с нашей моделью.
 *  Сначала мы определим переменную model. Затем мы создадим переменную SequentialBlock net, которая будет ссылаться
 *  на экземпляр класса SequentialBlock. Класс SequentialBlock определяет контейнер для нескольких слоев, которые будут
 *  объединены в цепочку. Получив входные данные, SequentialBlock передает их через первый слой, в свою очередь
 *  передавая выходные данные в качестве входных данных второго слоя и так далее. В следующем примере наша модель
 *  состоит только из одного слоя, поэтому нам на самом деле не нужен SequentialBlock. Но поскольку почти все наши
 *  будущие модели будут включать в себя несколько слоев, мы все равно будем использовать их, просто чтобы ознакомить
 *  вас с наиболее стандартным рабочим процессом.
 *
 * Слой называется полностью подключенным, поскольку каждый из его входов соединен с каждым из его выходов
 * посредством матрично-векторного умножения. В DJL мы можем использовать линейный блок для применения линейного
 * преобразования. Мы просто задаем количество выходных сигналов (в нашем случае оно равно 1) и выбираем,
 * хотим ли мы включить смещение (да).
 */

public static Model getModel(){
    Linear linearBlock= Linear.builder().optBias(true).setUnits(1).build();
    SequentialBlock net =new SequentialBlock();
    net.add(linearBlock);
    Model model = Model.newInstance("lin-Reg");
    model.setBlock(net);
    return model;
    }



}
