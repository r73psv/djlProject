package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;

import static com.github.r73pls.djl_Project.deepLearning.CustomBlock.translator;

public class NestMLP extends AbstractBlock {

    /**
     *С помощью DJL мы можем комбинировать различные способы сборки блоков друг с другом.
     * В следующем примере мы используем несколько креативных способов компоновки блоков.
     */

    private SequentialBlock net;
    private Block dense;

    private Block test;

    public NestMLP() {
        net = new SequentialBlock();
        net.add(Linear.builder().setUnits(64).build());
        net.add(Activation.reluBlock());
        net.add(Linear.builder().setUnits(32).build());
        net.add(Activation.reluBlock());
        addChildBlock("net", net);

        dense = addChildBlock("dense", Linear.builder().setUnits(16).build());
    }


    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> pairList) {
        NDList current = inputs;
        current = net.forward(parameterStore, current, training);
        current = dense.forward(parameterStore, current, training);
        current = new NDList(Activation.relu(current.singletonOrThrow()));
        return current;
    }


    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        Shape[] current = inputs;
        for (Block block : children.values()) {
            current = block.getOutputShapes(current);
        }
        return current;
    }
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape[] shapes = inputShapes;
        for (Block child : getChildren().values()) {
            child.initialize(manager, dataType, shapes);
            shapes = child.getOutputShapes(shapes);
        }
    }
    public void model(){
        NDManager manager=NDManager.newBaseManager();
        SequentialBlock chimera = new SequentialBlock();
        chimera.add(new NestMLP());
        chimera.add(Linear.builder().setUnits(20).build());
        chimera.add(new FixedHiddenMLP());

        chimera.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        chimera.initialize(manager, DataType.FLOAT32, CustomBlock.x.getShape());
        Model model = Model.newInstance("chimera");
        model.setBlock(chimera);
        NDList xList = new NDList(CustomBlock.x);
        Predictor predictor = model.newPredictor(translator);
        try {
            ((NDList) predictor.predict(xList)).singletonOrThrow();
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }
}
