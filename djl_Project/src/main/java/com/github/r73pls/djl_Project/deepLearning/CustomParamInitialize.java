package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.NormalInitializer;

import static ai.djl.nn.Activation.reluBlock;

public class CustomParamInitialize implements Initializer {

    /**
     *Иногда методы инициализации, которые нам нужны, не являются стандартными в DJL. В таких случаях мы можем
     * определить класс для реализации интерфейса инициализатора. Нам нужно только реализовать функцию initialize(),
     * которая принимает NDManager, форму и тип данных. Затем мы создаем NDArray с вышеупомянутыми формой
     * и типом данных и инициализируем его так, как мы хотим! Вы также можете настроить свой инициализатор так,
     * чтобы он принимал некоторые параметры. Просто объявите их как поля в классе и передайте в качестве входных
     * данных конструктору!
     * В примере ниже мы определяем инициализатор для следующего странного распределения:
     *     ⎧    U[5,10]     with probability 1/4
     * w∼  ⎨    0           with probability 1/2
     *     ⎩    U[−10,−5]   with probability 1/4
     */
    public CustomParamInitialize(){}
    @Override
    public NDArray initialize(NDManager ndManager, Shape shape, DataType dataType) {
        System.out.printf("Init %s\n",shape.toString());
        // Здесь мы генерируем точки данных из равномерного распределения [-10, 10]
        NDArray data =ndManager.randomUniform(-10,10,shape,dataType);
        // Мы сохраняем точки данных, абсолютное значение которых >= 5, а для остальных устанавливаем значение 0.
        // Это приводит к распределению "w", показанному выше.
        NDArray absGte5 = data.abs().gte(5);// возвращает логическое значение NDArray,
                                                    // где true указывает на abs >= 5 и false в противном случае
        return data.mul(absGte5);   // сохраняет индексы true и присваивает индексам false значение 0.
                                    // специальная операция при умножении числового значения NDArray на логическое значение NDArray
    }
        NDManager manager=NDManager.newBaseManager();
    SequentialBlock net = ParamInitialize.getNet();
    public void customInit () {
        net.setInitializer(new CustomParamInitialize(), Parameter.Type.WEIGHT);
        net.initialize(manager, DataType.FLOAT32, CustomBlock.x.getShape());
        Block linearLayer = net.getChildren().valueAt(0);
        NDArray weight = linearLayer.getParameters().valueAt(0).getArray();
        System.out.println(weight);
    }
    /**
     * Обратите внимание, что у нас всегда есть возможность задать параметры напрямую, вызвав GetValue().getArray()
     * для доступа к базовому NDArray. Примечание для опытных пользователей: вы не можете напрямую изменять параметры
     * в области GarbageCollector. Вы должны изменить их за пределами области действия GarbageCollector,
     * чтобы избежать путаницы в механизме автоматической дифференциации.
     */
    public void modifyParam(){
        // '__'i() - это операция на месте для изменения исходного NDArray
        NDArray weightLayer = net.getChildren().valueAt(0)
                .getParameters().valueAt(0).getArray();
        weightLayer.addi(7);
        weightLayer.divi(9);
        weightLayer.set(new NDIndex(0, 0), 2020); // set the (0, 0) index to 2020
        System.out.println(weightLayer);
    }
    /**
     * Часто мы хотим использовать общие параметры на нескольких уровнях. Например, при изучении встраивания
     * слов может оказаться разумным использовать одни и те же параметры как для кодирования,
     * так и для декодирования слов. Давайте посмотрим, как это сделать более элегантно.
     * Далее мы выделяем плотный слой, а затем используем его параметры специально для настройки параметров другого слоя.
     */
    public void tiedParameters (){
        SequentialBlock net = new SequentialBlock();
        // Нам нужно присвоить общему слою  такое имя, чтобы мы могли ссылаться на его параметры
        Block shared = Linear.builder().setUnits(8).build();
        SequentialBlock sharedRelu = new SequentialBlock();
        sharedRelu.add(shared);
        sharedRelu.add(Activation.reluBlock());
        net.add(Linear.builder().setUnits(8).build());
        net.add(Activation.reluBlock());
        net.add(sharedRelu);
        net.add(sharedRelu);
        net.add(Linear.builder().setUnits(10).build());

        NDArray x = manager.randomUniform(-10f,10f,new Shape(2,20),DataType.FLOAT32);
        net.setInitializer(new NormalInitializer(),Parameter.Type.WEIGHT);
        net.initialize(manager,DataType.FLOAT32, x.getShape());
        net.forward(new ParameterStore(manager, false),new NDList(x),false).singletonOrThrow();

    }
    /**
     * В этом примере показано, что параметры второго и третьего слоя связаны. Они не просто равны,
     * они представлены одним и тем же массивом данных. Таким образом, если мы изменим один из параметров,
     * другой тоже изменится. Вы можете задаться вопросом, что происходит с градиентами при привязке параметров?
     * Поскольку параметры модели содержат градиенты, градиенты второго скрытого слоя и третьего
     * скрытого слоя суммируются в shared.getGradient() во время обратного распространения.
     */

}
