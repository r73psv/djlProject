package com.github.r73pls.djl_Project.mlp;

public class ActivationFunction {

    /**
     * ReLU обеспечивает очень простое нелинейное преобразование. При заданном элементе z
     * функция определяется как максимальное значение этого элемента и 0.
     * ReLU(z)=max(z,0).
     * Неофициально, функция ReLU сохраняет только положительные элементы и отбрасывает все отрицательные (устанавливая
     * для соответствующих активаций значение 0). Чтобы получить представление, мы можем построить график функции.
     * Поскольку она используется так часто, DJL поддерживает функцию relu как собственный оператор.
     * Когда входные данные отрицательны, производная функции ReLU равна 0, а когда входные данные положительны,
     * производная функции ReLU равна 1. Обратите внимание, что функция ReLU не дифференцируема, когда входные данные
     * принимают значение, точно равное 0. В таких случаях мы по умолчанию используем левую производную (LHS) и говорим,
     * что производная равна 0, когда входные данные равны 0
     * Существует множество вариантов функции ReLU, включая параметризованную функцию Relu (PReLU) от He et al., 2015.
     * Этот вариант добавляет линейный член в ReLU, поэтому некоторая информация все равно передается, даже если аргумент
     * отрицательный.
     * PReLU(x)=max(0,x)+amin(0,x).
     * Причина использования Reality заключается в том, что его производные работают особенно хорошо: они либо исчезают,
     * либо просто пропускают аргумент. Это улучшает оптимизацию и устраняет хорошо документированную проблему
     * исчезновения градиентов, с которой сталкивались предыдущие версии нейронных сетей.
     */
}
