package com.github.r73pls.djl_Project.ndarray;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;
import tech.tablesaw.io.DataReader;
import tech.tablesaw.io.DataWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import tech.tablesaw.api.*;

public class DataPreprocessing {
    /**
     * 2.2. Предварительная обработка данных
     * Чтобы применить глубокое обучение к решению реальных задач, мы часто начинаем с предварительной обработки
     * необработанных данных, а не тех, которые хорошо подготовлены в формате NDArray. Среди популярных инструментов
     * анализа данных на Java обычно используется пакет table saw. Если вы использовали пакет pandas для Python, он
     * вам знаком.
     * В качестве примера, мы начнем с создания искусственного набора данных, который хранится в файле формата csv
     * (значения, разделенные запятыми) ../data/house_tiny.csv. Данные, хранящиеся в других форматах, могут
     * обрабатываться аналогичным образом.

     */
     //записываем набор данных построчно в csv-файл.

    String filePath= "\\resources\\";
    File file1= new File(filePath);
    boolean created= file1.mkdir();
    String dataFile = "\\resources\\house_tiny.csv";
    // Create file
    File f = new File(dataFile);
     boolean created1;

    {
        try {
            created1 = f.createNewFile();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

// Запись файла

    FileWriter writer;

    {
        try {
            writer = new FileWriter(dataFile);


        writer.write("NumRooms,Alley,Price\n"); //названия колонок
        writer.write("NA,Pave,127500\n");  //строки данных
        writer.write("2,NA,106000\n");
        writer.write("4,NA,178100\n");
        writer.write("NA,NA,140000\n");
        } catch (IOException e) {
        throw new RuntimeException(e);
        }
     }
    /**
     * Чтобы загрузить необработанный набор данных из созданного csv-файла, мы импортируем пакет tablesaw и вызываем
     * функцию read для чтения непосредственно из созданного нами csv-файла. Этот набор данных состоит из четырех
     * строк и трех столбцов, где каждая строка описывает количество комнат (“NumRooms”), тип переулка (“Alley”)
     * и цену (“Price”) дома.
     */

    Table data = Table.read().file("\\resources\\house_tiny.csv");
    /**
     *  Для обработки пропущенных данных типичными методами являются вычисление и удаление, при которых вычисление
     *  заменяет пропущенные значения замененными, в то время как удаление игнорирует пропущенные значения.
     *  Здесь мы рассмотрим вычисление.
     * Мы разделяем данные на входные и выходные, создавая новые таблицы и указывая желаемые столбцы, где в первой
     * используются первые два столбца, а во второй сохраняется только последний столбец. Если числовые значения
     * во входных данных отсутствуют, мы заменяем отсутствующие записи данных средним значением того же столбца.
     */

    Table inputs = data.create(data.columns()).removeColumns("Price");
    Table outputs = data.select("Price");

    Column column;

    {
        column = inputs.column("NumRooms");
        column.set(column.isMissing(),
                (int) inputs.nCol("NumRooms").mean());
    }

    /**
    * Для категориальных или дискретных значений во входных данных мы рассматриваем отсутствующие данные или null
     * как категорию. Поскольку столбец “Alley” принимает только два типа категориальных значений “Pave” и пустую строку,
     * которая представляет отсутствующие данные/null, tablesaw может автоматически преобразовать этот столбец
     * в два столбца. Мы изменим эти два столбца, чтобы присвоить им имена, которые будут “Alley_Pave” и “Alley_nan”.
     * Для строки с типом аллеи “Pave” значения “Alley_Pave” и “Alley_nan” будут равны true и false.
     * Для строки с отсутствующим типом аллеи будут установлены значения false и true. После этого мы добавим эти
     * столбцы в исходные данные/таблицу, но преобразуем их в double, чтобы значение true и false изменилось на 1 и 0
     * соответственно. Наконец, мы удаляем исходный столбец “Alley”.
     */

    StringColumn col = (StringColumn) inputs.column("Alley");
    List<BooleanColumn> dummies = col.getDummies();
   Table inp = inputs.removeColumns(col).addColumns(DoubleColumn.create("Alley_Pave", dummies.get(0).asDoubleArray()),
            DoubleColumn.create("Alley_nan", dummies.get(1).asDoubleArray()));

    /**
     * Преобразование в формат NDArray
     * Теперь, когда все записи во входных и выходных данных являются числовыми, их можно преобразовать в формат NDArray.
     * Как только данные будут представлены в этом формате, с ними можно будет дополнительно работать с помощью
     * функций NDArray.
     */

    NDManager nd = NDManager.newBaseManager();
    NDArray x = nd.create(inputs.as().doubleMatrix());

    /**
     * Вы можете представить себе вектор как простой список скалярных значений. Мы называем эти значения элементами
     * (записями или компонентами) вектора. Когда наши векторы представляют примеры из нашего набора данных, их значения
     * имеют некоторое реальное значение. Например, если бы мы обучали модель прогнозированию риска невозврата кредита,
     * мы могли бы связать каждого заявителя с вектором, компоненты которого соответствуют его доходу,
     * стажу работы, количеству предыдущих невозвратов и другим факторам и т.д. В математической нотации мы обычно
     * обозначаем векторы строчными буквами, выделенными жирным шрифтом (например, x, y и z).
     * Мы работаем с векторами с помощью одномерного массива данных. В общем случае NDArrays могут иметь
     * произвольную длину, в зависимости от ограничений памяти вашего компьютера.
     * Мы можем ссылаться на любой элемент вектора, используя индекс. Например, мы можем ссылаться на i-й элемент x
     * x.get(3)
     * Когда мы меняем местами строки и столбцы матрицы, результат называется транспонированием матрицы.
     * x.transpose()
     */
    NDArray B = x.duplicate(); // создание копии массива x


 }
