package com.github.r73pls.djl_Project.mlp;

import ai.djl.Model;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.initializer.NormalInitializer;

public class modelMlp {
    public static Model model(){
    Model model= Model.newInstance("Mlp");
    SequentialBlock net =new SequentialBlock();
    net.add(Blocks.batchFlattenBlock(784));
    net.add(Linear.builder().setUnits(256).build());
    net.add(Activation::relu);
    net.add(Linear.builder().setUnits(10).build());
    net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
    model.setBlock(net);
    return model;
    }
}
