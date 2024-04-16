package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.core.Linear;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.initializer.XavierInitializer;

public class ParamInitialize {
    /**
     * По умолчанию DJL инициализирует матрицы весов на основе заданного вами инициализатора, а все параметры
     * смещения имеют значение 0. Однако нам часто требуется инициализировать наши веса в соответствии с различными
     * другими протоколами. Пакет ai.djl.training.initializer от DJL предоставляет множество предустановленных методов
     * инициализации. Если мы хотим создать пользовательский инициализатор, нам нужно выполнить некоторую дополнительную работу.
     *
     * В DJL при установке инициализатора для блоков функция setInitializer() по умолчанию не перезаписывает никакие
     * предыдущие установленные инициализаторы. Таким образом, если вы установили инициализатор ранее, но решили,
     * что хотите изменить свой инициализатор, и снова вызвали setInitializer(), второй setInitializer() не перезапишет ваш первый.
     * Кроме того, когда вы вызываете setInitializer() для блока, все внутренние блоки также будут вызывать
     * setInitializer() с тем же заданным инициализатором. Это означает, что мы можем вызвать setInitializer() на самом
     * высоком уровне блока и знать, что для всех внутренних блоков, у которых еще не установлен инициализатор, будет
     * установлен этот данный инициализатор. Преимущество этой настройки в том, что нам не нужно беспокоиться о том,
     * что наш setInitializer() переопределит наши предыдущие инициализаторы для внутренних блоков!
     * Однако, если вы хотите, вы можете явно задать инициализатор для параметра, вызвав его функцию setInitializer() напрямую.
     * Давайте начнем с вызова встроенных инициализаторов. Приведенный ниже код инициализирует все параметры заданным
     * постоянным значением 1, используя инициализатор ConstantInitializer().
     * Обратите внимание, что в настоящее время это ничего не даст, поскольку мы уже установили наш инициализатор
     * в предыдущем блоке кода. Мы можем убедиться в этом, проверив вес параметра.
     */
SequentialBlock net= ParamAccess.sequential();
NDManager manager=NDManager.newBaseManager();
public void initParam(){
    net.setInitializer(new ConstantInitializer(1), Parameter.Type.WEIGHT);
    net.initialize(manager, DataType.FLOAT32, CustomBlock.x.getShape());
    Block linearLayer = net.getChildren().get(0).getValue();
    NDArray weight = linearLayer.getParameters().get(0).getValue().getArray();
    System.out.println(weight);
}
/*
Однако мы можем увидеть эти инициализации, если создадим новую сеть.
Давайте напишем функцию для удобного создания этих сетевых архитектур.
 */

    public SequentialBlock getNet() {
        SequentialBlock net = new SequentialBlock();
        net.add(Linear.builder().setUnits(8).build());
        net.add(Activation.reluBlock());
        net.add(Linear.builder().setUnits(1).build());
        return net;
    }

    /*
    Если мы запустим наш предыдущий инициализатор в этой новой сети и проверим параметр, мы увидим,
    что все инициализировано правильно! (для 7777!)
     */

    SequentialBlock net1 = getNet();
    public void initParam1(){
    net1.setInitializer(new ConstantInitializer(7777), Parameter.Type.WEIGHT);
    net1.initialize(manager, DataType.FLOAT32, CustomBlock.x.getShape());
    Block linearLayer = net.getChildren().valueAt(0);
    NDArray weight = linearLayer.getParameters().valueAt(0).getArray();
    System.out.println(weight);
    }
    /*
    Мы также можем инициализировать все параметры как гауссовские случайные величины со стандартным отклонением .01
     */

    public void initParamG(){
        net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        net.initialize(manager, DataType.FLOAT32, CustomBlock.x.getShape());
        Block linearLayer = net.getChildren().valueAt(0);
        NDArray weight = linearLayer.getParameters().valueAt(0).getArray();
        System.out.println(weight);
    }

  /*
    Мы также можем применить разные инициализаторы для определенных блоков. Например, ниже мы инициализируем первый слой
     с помощью инициализатора XavierInitializer, а второй слой - постоянным значением 0.Мы сделаем это
     без функции getNet(), так как будет проще иметь ссылку на каждый блок, который мы хотим установить.
   */

    public void initParamXI(){
        net = new SequentialBlock();
        Linear linear1 = Linear.builder().setUnits(8).build();
        net.add(linear1);
        net.add(Activation.reluBlock());
        Linear linear2 = Linear.builder().setUnits(1).build();
        net.add(linear2);

        linear1.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        linear1.initialize(manager, DataType.FLOAT32,CustomBlock.x.getShape());

        linear2.setInitializer(Initializer.ZEROS, Parameter.Type.WEIGHT);
        linear2.initialize(manager, DataType.FLOAT32, CustomBlock.x.getShape());

        System.out.println(linear1.getParameters().valueAt(0).getArray());
        System.out.println(linear2.getParameters().valueAt(0).getArray());
    }

    /*
    Наконец, мы можем напрямую получить доступ к параметру.setInitializer() и установить их инициализаторы по отдельности.
     */

    public void initParamIndividually (){
        net = getNet();
        ParameterList params = net.getParameters();

        params.get("01Linear_weight").setInitializer(new NormalInitializer());
        params.get("03Linear_weight").setInitializer(Initializer.ONES);

        net.initialize(manager, DataType.FLOAT32, new Shape(2, 4));

        System.out.println(params.valueAt(0).getArray());
        System.out.println(params.valueAt(2).getArray());
    }

}
