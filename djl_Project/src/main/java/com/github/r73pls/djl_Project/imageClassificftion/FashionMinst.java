package com.github.r73pls.djl_Project.imageClassificftion;

public class FashionMinst {
    /**
     * Изображения в Fashion-MNIST связаны со следующими категориями: футболка, брюки, пуловер, платье, пальто,
     * сандалии, рубашка, кроссовки, сумка и ботильоны. Следующая функция
     * преобразует числовые индексы этикеток в текстовые названия.
     */

    // Saved in the FashionMnist class for later use
    public String[] getFashionMnistLabels(int[] labelIndices) {
        String[] textLabels = {"футболка", "брюки", "полувер", "платье", "пальто",
                "сандалии", "рубашка", "кроссовки", "сумка", "ботильоны"};
        String[] convertedLabels = new String[labelIndices.length];
        for (int i = 0; i < labelIndices.length; i++) {
            convertedLabels[i] = textLabels[labelIndices[i]];
        }
        return convertedLabels;
    }

    public static String getFashionMnistLabel(int labelIndice) {
        String[] textLabels = {"футболка", "брюки", "полувер", "платье", "пальто",
                "сандалии", "рубашка", "кроссовки", "сумка", "ботильоны"};
        return textLabels[labelIndice];
    }

}
