package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.RNN;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.util.PairList;

public class CustomBlock extends AbstractBlock {

   static NDManager manager = NDManager.newBaseManager();
    static int inputSize = 20;
    static NDArray x = manager.randomUniform(0, 1, new Shape(2, inputSize)); // (2, 20) shape

    /**
     *  Основные функциональные возможности, которые должен обеспечивать каждый блок:
     * Используем входные данные в качестве аргументов для его метода forward().
     * Генерация выходных данных, используя функцию forward(), которая возвращает значение. Обратите внимание, что
     * выходные данные могут отличаться по форме от входных данных. Например, первый  слой в нашей модели,
     * принимает входные данные произвольного размера, но возвращает выходные данные размера 256.
     * Вычислите градиент его выходных данных по отношению к входным данным, к которым можно получить доступ
     * с помощью метода backward(). Обычно это происходит автоматически.
     * Сохраните и предоставьте доступ к параметрам, необходимым для выполнения вычисления forward().
     * Инициализируйте эти параметры по мере необходимости.
     *
     * В следующем фрагменте мы создаем блок с нуля, соответствующий многослойному персептрону с одним скрытым слоем
     * с 256 скрытыми узлами и 10-мерным выходным слоем. Обратите внимание, что этот класс,
     * наследует класс AbstractBlock. Мы будем в значительной степени полагаться на методы родительского класса,
     * а также реализовывать его обязательные для переопределения методы.
     */
    private static final byte VERSION = 1;
    private Block flattenInput;
    private Block hidden256;
    private Block output10;

    // Объявляем слой с параметрами модели. Здесь мы объявляем два полностью связанных слоя

    static Translator translator= new NoopTranslator();
    public CustomBlock(int inputSize){
        super(VERSION);
        flattenInput = addChildBlock("flattenInput", Blocks.batchFlattenBlock(inputSize)); //входной слой
        hidden256 = addChildBlock("hidden256", Linear.builder().setUnits(256).build()); // Скрытый слой
        output10= addChildBlock("output10", Linear.builder().setUnits(10).build()); //выходной слой

    }
    @Override
    // Определите прямое вычисление модели, то есть способ возврата требуемого вывода модели на основе входных данных x
    protected NDList forwardInternal(ParameterStore parameterStore, NDList input, boolean training, PairList<String, Object> pairList) {
        NDList current = input;
        current =flattenInput.forward(parameterStore,current,training);
        current =hidden256.forward(parameterStore,current,training);
        // Здесь мы используем функцию Activation.relu() поскольку она принимает NDArray, мы вызываем `singletonOrThrow()`
        //в NDList `current`, чтобы получить NDArray, а затем обернуть его в новый NDList для передачи к следующему вызову `forward()`
        current=new NDList(Activation.relu(current.singletonOrThrow()));
        current = output10.forward(parameterStore,current,training);
        return current;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        Shape [] current=inputs;
        for (Block block: children.values()){
            current= block.getOutputShapes(current);
        }
        return current;
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        hidden256.initialize(manager, dataType, new Shape(1, inputSize));
        output10.initialize(manager, dataType, new Shape(1, 256));
    }



    /**
     * Для начала давайте сосредоточимся на методе forward(). Обратите внимание, что в качестве входных данных он использует
     * ParameterStore, входные данные NDList, логическое обучение и параметры PairList<>, но сейчас  нужно позаботиться
     * только о входных данных NDList. Затем он передает входные данные через каждый уровень в этой реализации MLP,
     * оба уровня являются переменными экземпляра. Чтобы понять, почему это разумно, представьте, что вы создаете
     * два экземпляра MLP, net1 и net2, и обучаете их на разных данных. Естественно, мы ожидали бы, что они будут
     * представлять собой две разные усвоенные модели.
     *
     * Мы создаем экземпляры слоев MLP в методе initializeChildBlocks() и впоследствии вызываем эти слои при каждом вызове
     * метода forward(). Обратите внимание на несколько ключевых деталей. Во-первых, наш настроенный
     * метод initializeChildBlocks() вызывает метод initialize() каждого дочернего класса, что избавляет нас от
     * необходимости повторять стандартный код, применимый к большинству блоков. Затем мы создаем экземпляры наших двух
     * линейных слоев, добавляя их в hidden256 и output10. Обратите внимание, что если мы не реализуем новый оператор,
     * нам не нужно беспокоиться об обратном распространении (обратный метод) или инициализации параметра (метод initialize).
     * DJL сгенерирует эти методы автоматически. Нам также не нужно вызывать initializeChildBlocks(), а вместо этого
     * просто вызывать метод initialize() AbstractBlock, поскольку AbstractBlock автоматически
     * вызывает initializeChildBlocks() в нем. Давайте попробуем это сделать:
     */
public void  net(int inputSize){
    CustomBlock net=new CustomBlock(inputSize);
    net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
    net.initialize(manager,DataType.FLOAT32);
    Model model = Model.newInstance("fixed-mlp");
    model.setBlock(net);
    NDList xList = new NDList(x);
    Predictor predictor = model.newPredictor(translator);
    try {
        ((NDList) predictor.predict(xList)).singletonOrThrow();
    } catch (TranslateException e) {
        throw new RuntimeException(e);
    }

}
}
