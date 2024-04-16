package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.Model;
import ai.djl.inference.Predictor;
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

public class MySequential extends AbstractBlock {
    /**
     * Теперь мы можем более подробно рассмотреть, как работает класс Sequential Block. Напомним, что Sequential Block
     * был разработан для последовательного объединения других блоков. Чтобы создать наш собственный упрощенный
     * My Sequential, нам просто нужно определить два ключевых метода:
     * 1. Метод add() для добавления блоков один за другим в список.
     * 2. Метод forward() для передачи входных данных по цепочке блоков (в том же порядке, в каком они были добавлены).
     *
     * Нам нужно определить дополнительные вспомогательные методы:
     * 1. Метод initialize Child Blocks() для инициализации дочерних блоков.
     * 2. Метод getoutputshape() для возврата выходной формы.
     *
     * Класс MySequential обеспечивает ту же функциональность, что и класс SequentialBlock в DJL по умолчанию:
     */

    private static final byte VERSION = 2;

    public MySequential() {
        super(VERSION);
    }

    public MySequential add(Block block) {
        // Здесь block является экземпляром подкласса Block, и мы предполагаем, что он имеет уникальное имя.
        // Мы добавляем дочерний блок в список дочерних блоков с помощью `addChildBlock()`, который определен в AbstractBlock.
        if (block != null) {
            addChildBlock(block.getClass().getSimpleName(), block);
        }
        return this;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> pairList) {
        NDList current = inputs;
        for (Block block : children.values()) {
            // Блок-лист гарантирует, что участники будут просматриваться в том порядке, в котором они были добавлены
            current = block.forward(parameterStore, current, training);
        }
        return current;
    }
    @Override
    // Инициализирует все дочерние блоки
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape[] shapes = inputShapes;
        for (Block child : getChildren().values()) {
            child.initialize(manager, dataType, shapes);
            shapes = child.getOutputShapes(shapes);
        }
    }
    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        return inputs;
    }
    /**
     * Метод add() добавляет отдельный блок к дочерним элементам списка блоков, используя метод addChild Block(),
     * реализованный в AbstractBlock. Вы можете задаться вопросом, почему каждый блок DJL обладает атрибутом children
     * и почему мы использовали его, а не просто определили список Java самостоятельно. Короче говоря,
     * главное преимущество дочерних блоков заключается в том, что во время инициализации параметров нашего блока
     * DJL знает, что нужно посмотреть в списке дочерних блоков, чтобы найти подблоки, параметры которых также необходимо инициализировать.
     *
     * Когда вызывается метод forward() нашего блока MySequential, каждый добавленный блок выполняется в том порядке,
     * в котором они были добавлены. Теперь мы можем реализовать MLP, используя наш класс My Sequential.
     */
public void model(){
    NDManager manager=NDManager.newBaseManager();
    MySequential net = new MySequential();
    net.add(Linear.builder().setUnits(256).build());
    net.add(Activation.reluBlock());
    net.add(Linear.builder().setUnits(10).build());

    net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
    net.initialize(manager, DataType.FLOAT32, CustomBlock.x.getShape());

    Model model = Model.newInstance("my-sequential");
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

