package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.ParameterStore;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

public class MyLinear extends AbstractBlock {
    /**
     * Теперь, когда мы знаем, как определять простые слои, давайте перейдем к определению слоев с параметрами, которые
     * можно настроить в процессе обучения. Это позволяет нам указать DJL, для чего нам нужно вычислять градиенты.
     * Чтобы автоматизировать часть рутинной работы, Parameter class и ParameterList предоставляют некоторые базовые
     * функции для ведения домашнего хозяйства. В частности, они управляют доступом, инициализацией, совместным
     * использованием, сохранением и загрузкой параметров модели. Таким образом, помимо прочих преимуществ,
     * нам не нужно будет писать пользовательские процедуры сериализации для каждого пользовательского слоя.
     * Реализация собственной версии линейного слоя DJL. Напомним, что для этого слоя требуется два параметра:
     * один для веса, другой для смещения. В этой реализации мы используем активацию ReLU по умолчанию.
     * В конструкторе inUnits и outUnits обозначают количество входов и выходов соответственно.
     * Мы создаем экземпляр нового параметра, вызывая его конструктор и передавая имя, ссылку на блок,
     * с которым он должен быть связан, и его тип, который мы можем задать с помощью ParameterType.
     * Затем мы вызываем AddParameter() в нашем конструкторе Linear с новым созданным параметром и его соответствующей
     * формой. Мы делаем это как для веса, так и для смещения.
     */
    private Parameter weight;
    private Parameter bias;

    private int inUnits;
    private int outUnits;
    public MyLinear(int outUnits, int inUnits) {
        this.inUnits = inUnits;
        this.outUnits = outUnits;
        weight = addParameter(
                Parameter.builder()
                        .setName("weight")
                        .setType(Parameter.Type.WEIGHT)
                        .optShape(new Shape(inUnits, outUnits))
                        .build());
        bias = addParameter(
                Parameter.builder()
                        .setName("bias")
                        .setType(Parameter.Type.BIAS)
                        .optShape(new Shape(outUnits))
                        .build());
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Device device = input.getDevice();
        // Поскольку мы добавили этот параметр, теперь мы можем получить к нему доступ из хранилища параметров
        NDArray weightArr = parameterStore.getValue(weight, device, false);
        NDArray biasArr = parameterStore.getValue(bias, device, false);
        return relu(linear(input, weightArr, biasArr));
    }

    // Применяет линейное преобразование
    public static NDArray linear(NDArray input, NDArray weight, NDArray bias) {
        return input.dot(weight).add(bias);
    }

    // Применяет преобразование relu
    public static NDList relu(NDArray input) {
        return new NDList(Activation.relu(input));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        return new Shape[]{new Shape(outUnits, inUnits)};
    }

    NDManager manager =NDManager.newBaseManager();
    // Затем мы создаем экземпляр класса MyLinear и получаем доступ к его параметрам модели.
    MyLinear linear = new MyLinear(3, 5);
    public void getParam (){
        var params = linear.getParameters();
        for (Pair<String, Parameter> param : params) {
            System.out.println(param.getKey());
        }
    }
    // инициализируем и протестируем наш слой.
    NDArray input = manager.randomUniform(0, 1, new Shape(2, 5));
    public void testMyLinear() {
        linear.initialize(manager, DataType.FLOAT32, input.getShape());
        Model model = Model.newInstance("my-linear");
        model.setBlock(linear);
        Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator());
        try {
            predictor.predict(new NDList(input)).singletonOrThrow();
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }

    }
//Мы также можем создавать модели, используя пользовательские слои. Как только у нас это получится,
// мы сможем использовать это так же, как встроенный плотный слой.
public void createModel(){
    NDArray input = manager.randomUniform(0, 1, new Shape(2, 64));

    SequentialBlock net = new SequentialBlock();
    net.add(new MyLinear(8, 64)); // 64 units in -> 8 units out
    net.add(new MyLinear(1, 8)); // 8 units in -> 1 unit out
    net.initialize(manager, DataType.FLOAT32, input.getShape());

    Model model = Model.newInstance("lin-reg-custom");
    model.setBlock(net);

    Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator());
    try {
        predictor.predict(new NDList(input)).singletonOrThrow();
    } catch (TranslateException e) {
        throw new RuntimeException(e);
    }
}


}
