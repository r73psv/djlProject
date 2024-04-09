package com.github.r73pls.djl_Project.imageClassificftion;
import ai.djl.basicdataset.cv.classification.*;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;

import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Color;
import java.io.IOException;

public class DataSetImage {
    private static int batchSize = 256;
    private static boolean randomShuffle = true;
    NDManager manager= NDManager.newBaseManager();
    /**
     * сначала определим функцию getDataset(), которая получает и считывает набор данных Fashion-MNIST.
     * Она возвращает набор данных для обучающего набора или набора проверки, в зависимости от переданного в usage
     * атем вы можете вызвать getData(менеджер) для набора данных, чтобы получить соответствующий итератор.
     * Он также принимает значения BatchSize и randomShuffle, которые определяют размер каждого пакета и
     * необходимость случайной перетасовки данных соответственно.
     */

    public static void getDataSet() {
        FashionMnist mnistTrain = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();
        FashionMnist mnistTest = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();
        try {
            mnistTrain.prepare();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        try {
            mnistTest.prepare();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }

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

    /**
     * Теперь мы можем создать функцию для визуализации этих примеров. Не стоит слишком беспокоиться
     * о специфике визуализации. Это просто для того, чтобы помочь интуитивно понять данные. По сути,
     * мы считываем данные из нескольких точек и преобразуем их значение RGB из 0-255 в 0-1. Затем
     * мы устанавливаем цвет в оттенках серого и отображаем его вместе с надписями во внешнем окне.
     */

    // Saved in the FashionMnistUtils class for later use
    public static BufferedImage showImages(
         ArrayDataset dataset, int number, int width, int height, int scale, NDManager manager) {
        // Plot a list of images
        BufferedImage[] images = new BufferedImage[number];
        String[] labels = new String[number];
        for (int i = 0; i < number; i++) {
            Record record = dataset.get(manager, i);
            NDArray array = record.getData().get(0).squeeze(-1);
            int y = (int) record.getLabels().get(0).getFloat();
            images[i] = toImage(array, width, height);
            labels[i] = getFashionMnistLabel(y);
        }
        int w = images[0].getWidth() * scale;
        int h = images[0].getHeight() * scale;

        return ImageUtils.showImages(images, labels, w, h);
    }

    private static BufferedImage toImage(NDArray array, int width, int height) {
        System.setProperty("apple.awt.UIElement", "true");
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = (Graphics2D) img.getGraphics();
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                float c = array.getFloat(j, i) / 255; // scale down to between 0 and 1
                g.setColor(new Color(c, c, c)); // set as a gray color
                g.fillRect(i, j, 1, 1);
            }
        }
        g.dispose();
        return img;
    }
}