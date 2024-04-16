package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.*;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.NormalInitializer;


public class ParamAccess {
    /**
     * Доступ к параметрам
     * Параметры каждого слоя удобно хранить в паре<Строка, параметр>, состоящей из уникальной строки, которая служит
     * ключом для слоя, и самого параметра. ParameterList является расширением PairList и возвращается при вызове
     * метода getParameters() для блока. Мы можем проверить параметры сети, определенные выше. Когда модель определена
     * с помощью класса SequentialBlock, мы можем получить доступ к любой паре слоев<Строка, параметр>, вызвав функцию
     * get() в списке параметров и передав индекс нужного нам параметра. Вызов getKey() и GetValue()
     * для пары<Строка, параметр> приведет к получению имени и значения параметра соответственно.
     * Мы также можем напрямую получить нужный нам параметр из списка параметров, вызвав get() и передав его уникальный
     * ключ(строковую часть пары<String, Parameter>. Если мы вызовем valueAt() и передадим индекс, мы получим параметр напрямую.
     */

   static NDManager manager = NDManager.newBaseManager();
    public static SequentialBlock sequential() {
        SequentialBlock net = new SequentialBlock();
        net.add(Linear.builder().setUnits(8).build());
        net.add(Activation.reluBlock());
        net.add(Linear.builder().setUnits(1).build());
        net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        net.initialize(manager, DataType.FLOAT32, CustomBlock.x.getShape());
        return net;
    }
   static SequentialBlock net =sequential();
    ParameterStore ps = new ParameterStore(manager, false);
    ParameterList params = net.getParameters();
    public void printParam() {



        // Распечатайте все ключи (уникальные!)
        for (var pair : params) {
            System.out.println(pair.getKey());
        }

        // Используйте уникальный ключ для доступа к параметру
        NDArray dense0Weight = params.get("01Linear_weight").getArray();
        NDArray dense0Bias = params.get("01Linear_bias").getArray();

        // Используйте индексацию для доступа к параметру
        NDArray dense1Weight = params.valueAt(2).getArray();
        NDArray dense1Bias = params.valueAt(3).getArray();

        System.out.println(dense0Weight);
        System.out.println(dense0Bias);

        System.out.println(dense1Weight);
        System.out.println(dense1Bias);

    /**
     * В результате мы узнаем несколько важных вещей. Во-первых, каждый полностью подключенный слой имеет два параметра,
     * например, dense0Weight и dense0Bias, которые соответствуют весам и смещениям этого слоя соответственно.
     * Переменная params - это список параметров, который содержит пары ключ-значение для названия слоя и
     * параметр класса параметров. С помощью параметра мы можем получить базовые числовые значения в виде NDArrays,
     * вызвав для них getArray()! Как веса, так и смещения сохраняются в виде значений с плавающей точкой одинарной точности (FLOAT32).
     */

    /**
     * Целевые параметры
     * Параметры - это сложные объекты, содержащие данные, градиенты и дополнительную информацию. Поэтому нам необходимо
     * явно запрашивать данные. Обратите внимание, что вектор смещения состоит из нулей, поскольку мы не обновляли сеть
     * с момента ее инициализации.
     *
     * Обратите внимание, что в отличие от смещений, веса не равны нулю. Это связано с тем, что, в отличие от смещений,
     * веса инициализируются случайным образом. В дополнение к getArray(), каждый параметр также предоставляет метод
     * require Gradient(), который возвращает, нужны ли параметру градиенты для вычисления (который мы задаем
     * в NDArray с помощью attachGradient()). Градиент имеет ту же форму, что и вес. Чтобы на самом деле получить доступ
     * к градиенту, мы просто вызываем getGradient() в NDArray. Поскольку мы еще не вызывали обратное распространение
     * для этой сети, все его значения равны 0. Мы бы вызвали его, создав экземпляр GradientCollector
     * и выполнив наши вычисления внутри него.
     */
    dense0Weight.getGradient();
    }
    /**
     * Сбор параметров из вложенных блоков
     * Давайте посмотрим, как работают соглашения об именовании параметров, если мы вкладываем несколько блоков друг
     * в друга. Для этого мы сначала определяем функцию, которая создает блоки (так сказать, фабрику блоков),
     * а затем объединяем их в еще более крупные блоки.
     */

    public SequentialBlock block1() {
        SequentialBlock net = new SequentialBlock();
        net.add(Linear.builder().setUnits(32).build());
        net.add(Activation.reluBlock());
        net.add(Linear.builder().setUnits(16).build());
        net.add(Activation.reluBlock());
        return net;
    }

    public SequentialBlock block2() {
        SequentialBlock net = new SequentialBlock();
        for (int i = 0; i < 4; i++) {
            net.add(block1());
        }
        return net;
    }
public void rgnet(){

    SequentialBlock rgnet = new SequentialBlock();
    rgnet.add(block2());
    rgnet.add(Linear.builder().setUnits(10).build());
    rgnet.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
    rgnet.initialize(manager, DataType.FLOAT32, CustomBlock.x.getShape());
    rgnet.forward(ps, new NDList(CustomBlock.x), false).singletonOrThrow();

    /*
    Теперь, когда мы спроектировали сеть, давайте посмотрим, как она организована. Мы можем получить список именованных
    параметров, вызвав функцию getParameters(). Однако мы хотим видеть не только параметры, но и структуру нашей сети.
    Чтобы увидеть архитектуру нашей сети, мы можем просто распечатать блок, архитектуру которого мы хотим увидеть.
     */
    System.out.println(rgnet);

    /**
     * SequentialBlock(2, 4) {
     *     SequentialBlock(2, 4) {
     *             SequentialBlock(2, 4) {
     *                     Linear(2, 4) -> (2, 32)
     *                     ReLU(2, 32) -> (2, 32)
     *                     Linear(2, 32) -> (2, 16)
     *                     ReLU(2, 16) -> (2, 16)
     *             } -> (2, 16)
     *             SequentialBlock(2, 16) {
     *                     Linear(2, 16) -> (2, 32)
     *                     ReLU(2, 32) -> (2, 32)
     *                     Linear(2, 32) -> (2, 16)
     *                     ReLU(2, 16) -> (2, 16)
     *             } -> (2, 16)
     *             SequentialBlock(2, 16) {
     *                     Linear(2, 16) -> (2, 32)
     *                     ReLU(2, 32) -> (2, 32)
     *                     Linear(2, 32) -> (2, 16)
     *                     ReLU(2, 16) -> (2, 16)
     *             } -> (2, 16)
     *             SequentialBlock(2, 16) {
     *                     Linear(2, 16) -> (2, 32)
     *                     ReLU(2, 32) -> (2, 32)
     *                     Linear(2, 32) -> (2, 16)
     *                     ReLU(2, 16) -> (2, 16)
     *             } -> (2, 16)
     *     } -> (2, 16)
     *     Linear(2, 16) -> (2, 10)
     * } -> (2, 10)
     */

    for (var param : rgnet.getParameters()) {
        System.out.println(param.getValue().getArray());
    }


/**
 * Поскольку слои иерархически вложены, мы также можем получить к ним доступ, вызвав их метод getChildren(),
 * чтобы получить список блоков (также являющийся расширением PairList) их внутренних блоков. Он использует методы
 * совместно со списком параметров, и поэтому мы можем использовать их знакомую структуру для доступа к блокам.
 * Мы можем вызвать get(i), чтобы получить пару<Строка, блок> с нужным нам индексом i, а затем, наконец, GetValue(),
 * чтобы получить фактический блок. Мы можем сделать это за один шаг, как показано выше, с помощью valueAt(i).
 * Затем мы должны повторить это, чтобы получить дочерний блок и так далее.
 * Здесь мы получаем доступ к первому основному блоку, внутри него - ко второму подблоку,
 * а внутри него - к смещению первого слоя следующим образом:
 */

    Block majorBlock1 = rgnet.getChildren().get(0).getValue();
    Block subBlock2 = majorBlock1.getChildren().valueAt(1);
    Block linearLayer1 = subBlock2.getChildren().valueAt(0);
    NDArray bias = linearLayer1.getParameters().valueAt(1).getArray();
    System.out.println(bias);

    }
}
