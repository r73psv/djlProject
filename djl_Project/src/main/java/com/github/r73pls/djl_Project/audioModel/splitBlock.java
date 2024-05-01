package com.github.r73pls.djl_Project.audioModel;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.MultiBoxPrior;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
public final class splitBlock extends AbstractBlock {
    private static final byte VERSION = 2;

    private List<Block> features;
    private List<Block> classPredictionBlocks;
    private List<Block> anchorPredictionBlocks;

    private List<MultiBoxPrior> multiBoxPriors;
    private int numClasses;
    NDManager manager = NDManager.newBaseManager();
    private splitBlock(Builder builder){
        super(VERSION);
        features = builder.features;
        features.forEach((block) -> addChildBlock(block.getClass().getSimpleName(), block));
        numClasses = builder.numClasses;
        classPredictionBlocks = builder.classPredictionBlocks;
        classPredictionBlocks.forEach(
                (block) -> addChildBlock(block.getClass().getSimpleName(), block));
        anchorPredictionBlocks = builder.anchorPredictionBlocks;
        anchorPredictionBlocks.forEach(
                (block) -> addChildBlock(block.getClass().getSimpleName(), block));
        multiBoxPriors = builder.multiBoxPriors;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore,
                                     NDList inputs, boolean training,
                                     PairList<String, Object> pairList) {
        NDList networkOutput = inputs;
        NDArray[] anchorsOutputs = new NDArray[features.size()];
        NDArray[] classOutputs = new NDArray[features.size()];
        NDArray[] boundingBoxOutputs = new NDArray[features.size()];
        for (int i = 0; i < features.size(); i++) {
            networkOutput = features.get(i).forward(parameterStore, networkOutput, training);

            MultiBoxPrior multiBoxPrior = multiBoxPriors.get(i);

            anchorsOutputs[i] = multiBoxPrior.generateAnchorBoxes(networkOutput.singletonOrThrow());
            classOutputs[i] =
                    classPredictionBlocks
                            .get(i)
                            .forward(parameterStore, networkOutput, training)
                            .singletonOrThrow();
            boundingBoxOutputs[i] =
                    anchorPredictionBlocks
                            .get(i)
                            .forward(parameterStore, networkOutput, training)
                            .singletonOrThrow();
        }
        NDArray anchors = NDArrays.concat(new NDList(anchorsOutputs), 1);
        NDArray classPredictions = concatPredictions(new NDList(classOutputs));
        NDArray boundingBoxPredictions = concatPredictions(new NDList(boundingBoxOutputs));
        classPredictions = classPredictions.reshape(classPredictions.size(0), -1, numClasses + 1);

        return new NDList(anchors, classPredictions, boundingBoxPredictions);
    }

    private NDArray concatPredictions(NDList output) {
        // transpose and batch flatten
        NDArray[] flattenOutput =
                output.stream()
                        .map(array -> array.transpose(0, 2, 3, 1).reshape(array.size(0), -1))
                        .toArray(NDArray[]::new);
        return NDArrays.concat(new NDList(flattenOutput), 1);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        try (NDManager manager = NDManager.newBaseManager()) {
            // TODO: output shape is wrong
            Shape[] childInputShapes = inputShapes;
            Shape[] anchorShapes = new Shape[features.size()];
            Shape[] classPredictionShapes = new Shape[features.size()];
            Shape[] anchorPredictionShapes = new Shape[features.size()];
            for (int i = 0; i < features.size(); i++) {
                childInputShapes = features.get(i).getOutputShapes(childInputShapes);
                anchorShapes[i] =
                        multiBoxPriors
                                .get(i)
                                .generateAnchorBoxes(manager.ones(childInputShapes[0]))
                                .getShape();
                classPredictionShapes[i] =
                        classPredictionBlocks.get(i).getOutputShapes(childInputShapes)[0];
                anchorPredictionShapes[i] =
                        anchorPredictionBlocks.get(i).getOutputShapes(childInputShapes)[0];
            }
            Shape anchorOutputShape = new Shape();
            for (Shape shape : anchorShapes) {
                anchorOutputShape = concatShape(anchorOutputShape, shape, 1);
            }

            NDList classPredictions = new NDList();
            for (Shape shape : classPredictionShapes) {
                classPredictions.add(manager.ones(shape));
            }
            NDArray classPredictionOutput = concatPredictions(classPredictions);
            Shape classPredictionOutputShape =
                    classPredictionOutput
                            .reshape(classPredictionOutput.size(0), -1, numClasses + 1)
                            .getShape();
            NDList anchorPredictions = new NDList();
            for (Shape shape : anchorPredictionShapes) {
                anchorPredictions.add(manager.ones(shape));
            }
            Shape anchorPredictionOutputShape = concatPredictions(anchorPredictions).getShape();
            return new Shape[] {
                    anchorOutputShape, classPredictionOutputShape, anchorPredictionOutputShape
            };
        }
    }
    private Shape concatShape(Shape shape, Shape concat, int axis) {
        if (shape.dimension() == 0) {
            return concat;
        }
        if (shape.dimension() != concat.dimension()) {
            throw new IllegalArgumentException("Shapes must have same dimensions");
        }
        long[] dimensions = new long[shape.dimension()];
        for (int i = 0; i < shape.dimension(); i++) {
            if (axis == i) {
                dimensions[i] = shape.get(i) + concat.get(i);
            } else {
                if (shape.get(i) != concat.get(i)) {
                    throw new UnsupportedOperationException(
                            "These shapes cannot be concatenated along axis " + i);
                }
                dimensions[i] = shape.get(i);
            }
        }
        return new Shape(dimensions);
    }
    public static SequentialBlock getSplitBlock() {
        SequentialBlock sequentialBlock = new SequentialBlock();
        sequentialBlock
                .add(Conv2d.builder()
                         .setKernelShape(new Shape(3,3))
                         .setFilters(32)
                         .optStride(new Shape(1,1))
                         .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3,3))
                        .setFilters(16)
                        .optStride(new Shape(1,1))
                        .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3)))
                .add(Dropout
                        .builder()
                        .optRate(0.25f)
                        .build())
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3,3))
                        .setFilters(64)
                        .optStride(new Shape(1,1))
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3,3))
                        .setFilters(16)
                        .optStride(new Shape(1,1))
                        .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3)))
                .add(Dropout
                        .builder()
                        .optRate(0.5f)
                        .build())
                .add(Blocks.batchFlattenBlock())
                .add(Linear
                        .builder()
                        .setUnits(128)
                        .build())
                .add(Activation::relu)
                .add(Dropout
                        .builder()
                        .optRate(0.5f)
                        .build())
                .add(Linear
                        .builder()
                        .setUnits(513)
                        .build());
        return sequentialBlock;
    }

    /**
     * Создает конструктор для построения {@link splitBlock}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** Строитель, чтобы построить {@link splitBlock}. */
    public static class Builder {

        private Block network;
        private int numFeatures = -1;
        private List<Block> features;
        private List<List<Float>> sizes;
        private List<List<Float>> ratios;
        private List<Block> classPredictionBlocks = new ArrayList<>();
        private List<Block> anchorPredictionBlocks = new ArrayList<>();
        private List<MultiBoxPrior> multiBoxPriors = new ArrayList<>();
        private int numClasses;
        private boolean globalPool = true;

        Builder() {
        }
    }


    }
