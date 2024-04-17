package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class CenteredLayer extends AbstractBlock {
    /**
     * Для начала мы создадим пользовательский слой (блок), который не имеет собственных параметров.
     *  Класс CenteredLayer просто вычитает среднее значение из своих входных данных. Чтобы создать его,
     *  нам просто нужно унаследовать от класса AbstractBlock и реализовать методы forward() и getOutputShapes().
     */
    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> pairList) {
        NDList current = inputs;
        return new NDList(current.head().sub(current.head().mean()));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        return inputs;
    }

}
