package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;

public class FixedHiddenMLP extends AbstractBlock {

    /**
     * Класс SequentialBlock упрощает построение модели, позволяя нам создавать новые архитектуры без необходимости
     * определять собственный класс. Однако не все архитектуры являются простыми последовательными цепочками.
     * Когда потребуется большая гибкость, мы захотим определить наши собственные блоки.
     * Например, мы могли бы захотеть выполнить поток управления Java в рамках метода forward.
     * Более того, мы могли бы захотеть выполнять произвольные математические операции, а не просто полагаться
     * на предопределенные уровни нейронной сети.До сих пор все операции в наших сетях выполнялись в соответствии
     * с активациями нашей сети и ее параметрами. Однако иногда нам может потребоваться включить условия, которые
     * не являются ни результатом предыдущих уровней, ни обновляемыми параметрами.
     * В DJL мы называем эти параметры постоянными. Скажем, например, что нам нужен слой, который вычисляет функцию
     * f(x,w)=c⋅w⊤x, где x является входным сигналом, w является нашим параметром, и c это некоторая заданная константа,
     * которая не обновляется во время оптимизации.
     * В следующем коде мы реализуем модель, которую нелегко собрать, используя только предопределенные слои и SequentialBlock.
     */

    private static final byte VERSION = 1;

    private Block hidden20;
    private NDArray constantParamWeight;
    private NDArray constantParamBias;

    public FixedHiddenMLP() {
        super(VERSION);
        hidden20 = addChildBlock("denseLayer", Linear.builder().setUnits(20).build());
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> pairList) {
        NDList current = inputs;

        // Полностью подключенный слой
        current = hidden20.forward(parameterStore, current, training);
        // Используйте постоянные параметры NDArray. Вызовите внутренний метод NDArray `linear()` для выполнения вычисления
        current = Linear.linear(current.singletonOrThrow(), constantParamWeight, constantParamBias);
        // Relu Activation
        current = new NDList(Activation.relu(current.singletonOrThrow()));
        // Повторно используйте полностью подключенный слой. Это эквивалентно совместному использованию
        // параметров двумя полностью подключенными слоями
        current = hidden20.forward(parameterStore, current, training);

        // Здесь, в потоке управления, мы возвращаем скаляр для сравнения
        while (current.head().abs().sum().getFloat() > 1) {
            current.head().divi(2);
        }
        return new NDList(current.head().abs().sum());
    }
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape[] shapes = inputShapes;
        for (Block child : getChildren().values()) {
            child.initialize(manager, dataType, shapes);
            shapes = child.getOutputShapes(shapes);
        }
        // Инициализировать уровень постоянных параметров
        constantParamWeight = manager.randomUniform(-0.07f, 0.07f, new Shape(20, 20));
        constantParamBias = manager.zeros(new Shape(20));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        return new Shape[]{new Shape(1)}; // мы возвращаем скаляр, чтобы форма была равна 1
    }

    /**
     * В этой фиксированной скрытой модели MLP мы реализуем скрытый слой, веса которого случайным образом
     * инициализируются при создании экземпляра и в дальнейшем остаются постоянными. Этот вес не является параметром
     * модели и, следовательно, никогда не обновляется при обратном распространении. Затем сеть передает выходные данные
     * этого фиксированного слоя через линейный слой.
     *
     * Обратите внимание, что перед возвратом выходных данных наша модель выполнила нечто необычное. Мы запустили цикл
     * while, проверив условие np.abs(x).sum() > 1 и разделив наш выходной вектор на 2  пока не будет выполнено условие.
     * Наконец, мы вернули сумму записей в x. Насколько нам известно, ни одна стандартная нейронная сеть не выполняет
     * эту операцию. Обратите внимание, что эта конкретная операция может оказаться бесполезной ни в одной реальной задаче.
     * Наша цель состоит только в том, чтобы показать вам, как интегрировать произвольный код в поток вычислений вашей нейронной сети.
     */

    public void model(){
        NDManager manager=NDManager.newBaseManager();
        FixedHiddenMLP net = new FixedHiddenMLP();

        net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        net.initialize(manager, DataType.FLOAT32, CustomBlock.x.getShape());

        Model model = Model.newInstance("fixed-mlp");
        model.setBlock(net);
        NDList xList = new NDList(CustomBlock.x);
        Predictor predictor = model.newPredictor(CustomBlock.translator);
        try {
            ((NDList) predictor.predict(xList)).singletonOrThrow();
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }
}
