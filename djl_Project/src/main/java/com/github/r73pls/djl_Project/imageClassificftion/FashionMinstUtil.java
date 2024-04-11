package com.github.r73pls.djl_Project.imageClassificftion;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Record;

import java.awt.*;
import java.awt.image.BufferedImage;

public class FashionMinstUtil {

    /**
     * Теперь мы можем создать функцию для визуализации этих примеров. Не стоит слишком беспокоиться
     * о специфике визуализации. Это просто для того, чтобы помочь интуитивно понять данные. По сути,
     * мы считываем данные из нескольких точек и преобразуем их значение RGB из 0-255 в 0-1. Затем
     * мы устанавливаем цвет в оттенках серого и отображаем его вместе с надписями во внешнем окне.
     */

     //Сохраняется в классе Fashion Mnist Utils для дальнейшего использования.
//    public static BufferedImage showImages(
//            ArrayDataset dataset, int number, int width, int height, int scale, NDManager manager) {
//        // Plot a list of images
//        BufferedImage[] images = new BufferedImage[number];
//        String[] labels = new String[number];
//        for (int i = 0; i < number; i++) {
//            Record record = dataset.get(manager, i);
//            NDArray array = record.getData().get(0).squeeze(-1);
//            int y = (int) record.getLabels().get(0).getFloat();
//            images[i] = toImage(array, width, height);
//            labels[i] = FashionMinst.getFashionMnistLabel(y);
//        }
//        int w = images[0].getWidth() * scale;
//        int h = images[0].getHeight() * scale;
//
//        return ImageUtils.showImages(images, labels, w, h);
//    }

    private static BufferedImage toImage(NDArray array, int width, int height) {
        System.setProperty("apple.awt.UIElement", "true");
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = (Graphics2D) img.getGraphics();
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                float c = array.getFloat(j, i) / 255; // масштабировать до значения от 0 до 1
                g.setColor(new Color(c, c, c)); // установить серый цвет
                g.fillRect(i, j, 1, 1);
            }
        }
        g.dispose();
        return img;
    }
}
