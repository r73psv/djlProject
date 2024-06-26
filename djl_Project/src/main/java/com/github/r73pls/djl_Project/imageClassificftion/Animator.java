package com.github.r73pls.djl_Project.imageClassificftion;
import tech.tablesaw.api.*;
import tech.tablesaw.api.Row;
import tech.tablesaw.columns.Column;
import ai.djl.training.dataset.Record;
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Color;
import java.awt.DisplayMode.*;
import tech.tablesaw.api.FloatColumn;

import tech.tablesaw.plotly.api.LinePlot;

import tech.tablesaw.plotly.api.*;
import org.apache.commons.lang3.ArrayUtils;

public class Animator {
    private String id; // Id reference of graph(for updating graph)
    private Table data; // Data Points

    public Animator() {
        id = "";

        // Incrementally plot data
        data = Table.create("Data")
                .addColumns(
                        FloatColumn.create("epoch", new float[]{}),
                        FloatColumn.create("value", new float[]{}),
                        StringColumn.create("metric", new String[]{})
                );
    }

    // Add a single metric to the table
    public void add(float epoch, float value, String metric) {
        Row newRow = data.appendRow();
        newRow.setFloat("epoch", epoch);
        newRow.setFloat("value", value);
        newRow.setString("metric", metric);
    }

    // Добавьте показатели точности, точности поездов и потерь поездов для данной эпохи.
    // Then plot it on the graph
    public void add(float epoch, float accuracy, float trainAcc, float trainLoss) {
        add(epoch, trainLoss, "train loss");
        add(epoch, trainAcc, "train accuracy");
        add(epoch, accuracy, "test accuracy");
//        show();
    }

    // Отображение графика
//    public void show() {
//        if (id.equals("")) {
//           id = display(LinePlot.create("", data, "epoch", "value", "metric"));
//            return;
//        }
//        update();
//    }

    //Обновить график
//    public void update() {
//       updateDisplay(id, LinePlot.create("", data, "epoch", "value", "metric"));
//    }

    // Returns the column at the given index
    // if it is a float column
    // Otherwise returns null
    public float[] getY(Table t, int index) {
        Column c = t.column(index);
        if (c.type() == ColumnType.FLOAT) {
            float[] newArray = new float[c.size()];
            System.arraycopy(c.asList().toArray(), 0, newArray, 0, c.size());
            return newArray;
        }
        return null;
    }

}
