package com.github.r73pls.djl_Project.deepLearning;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class LoadSaveModel {
    /**
     * Сохранение отдельных весовых векторов (или других тензоров) полезно, но становится очень утомительным,
     * если мы хотим сохранить (а затем загрузить) всю модель целиком. В конце концов, у нас могут быть сотни
     * разбросанных групп параметров. По этой причине платформа предоставляет встроенную функциональность для загрузки
     * и сохранения целых сетей. Важно отметить, что при этом сохраняются параметры модели, а не вся модель целиком.
     * Например, если у нас трехуровневый MLP, нам нужно указать архитектуру отдельно. Причина этого в том,
     * что сами модели могут содержать произвольный код, следовательно, они не могут быть сериализованы
     * естественным образом. Таким образом, чтобы восстановить модель, нам нужно сгенерировать архитектуру в коде,
     * а затем загрузить параметры с диска. Давайте начнем с нашего знакомого MLP.
     */
    NDManager manager=NDManager.newBaseManager();
    public SequentialBlock createMLP() {
        SequentialBlock mlp = new SequentialBlock();
        mlp.add(Linear.builder().setUnits(256).build());
        mlp.add(Activation.reluBlock());
        mlp.add(Linear.builder().setUnits(10).build());
        return mlp;
    }
//Далее мы сохраняем параметры модели в виде файла с именем mlp.param.
public void saveParam() throws IOException {
    SequentialBlock original = createMLP();
    NDArray x = manager.randomUniform(0, 1, new Shape(2, 5));
    original.initialize(manager, DataType.FLOAT32, x.getShape());

    ParameterStore ps = new ParameterStore(manager, false);
    NDArray y = original.forward(ps, new NDList(x), false).singletonOrThrow();
    // Save file
    File mlpParamFile = new File("mlp.param");
    DataOutputStream os = new DataOutputStream(Files.newOutputStream(mlpParamFile.toPath()));
    original.saveParameters(os);
}
//Чтобы восстановить модель, мы создаем экземпляр исходной MLP-модели.
// Вместо случайной инициализации параметров модели мы считываем параметры, сохраненные в файле, напрямую.
    public void loadParam(File mlpParamFile) throws IOException, MalformedModelException {
        // Create duplicate of network architecture
        SequentialBlock clone = createMLP();
        // Load Parameters
        clone.loadParameters(manager, new DataInputStream(Files.newInputStream(mlpParamFile.toPath())));
    }

}
