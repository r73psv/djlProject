package com.github.r73pls.djl_Project.imageClassificftion;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.dataset.Batch;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collector;
import java.util.concurrent.atomic.DoubleAccumulator;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;
import java.util.function.BinaryOperator;
public class LossFunction {
   private static NDManager manager =NDManager.newBaseManager();
    /**
     * Нам нужно реализовать функцию кросс-энтропийных потерь. Это, возможно, самая распространенная функция потерь
     * во всем глубоком обучении.
     * Перекрестная энтропия принимает отрицательную логарифмическую вероятность прогнозируемой вероятности,
     * присвоенной истинной метке -log P(y mid x). Вместо того, чтобы повторять предсказания с помощью цикла Java
     * for (который, как правило, неэффективен), мы можем использовать функцию NDArray get() в сочетании с NDIndex,
     * что позволит нам легко выбирать подходящие термины из матрицы записей softmax.
     * В других фреймворках, таких как PyTorch, это обычно называется оператором pick().
     *В разделе ":, {}" NDIndex выбираются все массивы, а manager.create(new int[]{0, 2}) создает NDArray
     * со значениями 0 и 2, чтобы выбрать 0-й и 2-й элементы для каждого соответствующего NDArray.
     * Примечание: при использовании NDIndex таким образом, передаваемый в NDArray, используемый для выбора индексов,
     * должен иметь тип int или long. Вы можете использовать функцию toType(), чтобы изменить тип NDArray,
     * который будет показан ниже.
     */
    public static NDArray getLossFunction(NDArray yHat, NDArray y){
        //NDArray yHat = manager.create(new float[][]{{0.1f, 0.3f, 0.6f}, {0.3f, 0.2f, 0.5f}});
        //yHat.get(new NDIndex(":, {}", manager.create(new int[]{0, 2})));
        // Здесь необязательно, чтобы y имел тип данных int или long
        // и в нашем случае мы знаем, что это значение float32.
        // Сначала мы должны преобразовать его в int или long(здесь мы выбираем int).
        // прежде чем мы сможем использовать его с помощью NDIndex для "выбора" индексов.
        // Он также принимает логическое значение для возврата копии существующего NDArray
        // но мы этого не хотим, поэтому мы передаем `false`
        NDIndex pikIndex=new NDIndex()
                .addAllDim(Math.floorMod(-1,yHat.getShape().dimension()))
                .addPickDim(y);
        return yHat.get(pikIndex).log().neg();
    }
    /**
     *Учитывая предсказанное распределение вероятностей, мы обычно выбираем класс с наибольшей предсказанной
     * вероятностью, когда нам нужно получить точный прогноз.
     * Если прогнозы соответствуют фактической категории y, они верны. Точность классификации - это доля всех правильных
     * прогнозов. Хотя напрямую оптимизировать точность может быть сложно (она не поддается дифференциации), часто нас
     * больше всего волнует именно показатель производительности, и мы почти всегда сообщаем о нем при обучении
     * классификаторов. Чтобы вычислить точность, мы делаем следующее:
     * сначала мы выполняем yHat.argMax(1), где 1 - ось для сбора прогнозируемых классов (заданных индексами
     * для наибольших записей в каждой строке). Результат имеет ту же форму, что и переменная y. Теперь нам просто
     * нужно проверить, насколько часто эти два параметра совпадают. Поскольку функция равенства eq() чувствительна
     * к типу данных (например, float32 и float32 никогда не бывают равны), нам также нужно преобразовать их в один
     * и тот же тип (мы выбираем int32). Результатом является NDArray, содержащий записи 0 (ложь) и 1 (истина).
     * Затем мы суммируем количество правильных записей и преобразуем результат в число с плавающей запятой.
     * Наконец, мы получаем среднее значение путем деления на количество точек данных.
     */

     public static float accuracy (NDArray yHat, NDArray y){
        // Проверяеь размер 1-го измерения, чтобы увидеть, есть ли у нас несколько образцов
        if (yHat.getShape().size(1) > 1) {
            // Argmax получает индекс максимального числа аргументов для данной оси 1
            // Преобразует yHat в тот же тип данных, что и y (int32)
            // Подсчитывает количество истинных записей
            return yHat.argMax(1).toType(DataType.INT32, false).eq(y.toType(DataType.INT32, false))
                    .sum().toType(DataType.FLOAT32, false).getFloat();
        }
            return yHat.toType(DataType.INT32, false).eq(y.toType(DataType.INT32, false))
                    .sum().toType(DataType.FLOAT32, false).getFloat();

    }
    public static float evaluateAccuracy(UnaryOperator <NDArray> net, Iterable <Batch> dataIterator) {
        Accumulator metric = new Accumulator (2);  // numCorrectedExamples, numExamples
        Batch batch = dataIterator.iterator().next();
        NDArray X = batch.getData().head();
        NDArray y = batch.getLabels().head();
        metric.add(new float[]{accuracy(net.apply(X), y), (float)y.size()});
        batch.close();

        return metric.get(0) / metric.get(1);
    }

}

