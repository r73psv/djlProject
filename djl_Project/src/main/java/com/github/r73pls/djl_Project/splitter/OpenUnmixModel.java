package com.github.r73pls.djl_Project.splitter;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.audio.AudioUtils;
import ai.djl.modality.audio.FloatAudioBuffer;
import ai.djl.ndarray.NDArray;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.io.File;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import splitter.AudioUtils;
public class OpenUnmixModel {

    public static void main(String[] args) {
        try {
            Model model = ZooModel.newInstance("ai/djl/open_unmix").loadModel();

            File audioFile = new File("path/to/audio/file.wav");
            FloatAudioBuffer audioBuffer = AudioUtils.readAudio(audioFile);

            try (Translator<FloatAudioBuffer, NDArray> translator = model.newTranslator(new TranslatorContext())) {
                try (Predictor<FloatAudioBuffer, NDArray> predictor = model.newPredictor(translator)) {
                    NDArray components = predictor.predict(audioBuffer);

                    //  сохраняем компоненты музыки
                    saveAudioComponents(components, "outputPath");
                }
            }
        } catch (ModelNotFoundException | IOException e) {
            e.printStackTrace();
        }
    }

    // Метод для сохранения компонентов музыки в отдельные файлы
    public static void saveAudioComponents(NDArray components, String outputPath) {
        NDList componentList = components.split(4, 1);

        for (int i = 0; i < componentList.size(); i++) {
            NDArray component = componentList.get(i);
            String componentName;
            switch (i) {
                case 0:
                    componentName = "vocals";
                    break;
                case 1:
                    componentName = "drums";
                    break;
                case 2:
                    componentName = "bass";
                    break;
                case 3:
                    componentName = "other";
                    break;
                default:
                    componentName = "component" + i;
            }
            saveAudioFile(component, outputPath + "/" + componentName + ".wav");
        }
    }

    private static void saveAudioFile(NDArray audioData, String outputFilePath) {
        try {
            float[] audioFloatArray = audioData.toFloatArray();
            float sampleRate = 44100; // Sample rate for audio data

            byte[] audioByteArray = AudioUtils.toData(audioFloatArray, sampleRate);
            AudioUtils.save(outputFilePath, audioByteArray);
            System.out.println("Saved component to: " + outputFilePath);
        } catch (Exception e) {
            System.err.println("Error saving audio file: " + e.getMessage());
        }
    }

    /**
     *  загружаем модель "open-unmix" из хранилища Zoo Model с помощью
     * ZooModel.newInstance("ai/djl/open_unmix").loadModel() и используем ее для разделения компонентов входной аудио
     * записи на вокал, басы, ударные и другие.
     * Затем мы предварительно обрабатываем аудиофайл и делаем предсказания компонентов музыки с помощью
     * model.newPredictor(translator).predict(audioBuffer).
     * Наконец, результаты предсказаний (компоненты музыки) могут быть обработаны (например, сохранены в отдельные файлы)
     * с помощью метода saveComponents().
     * Пожалуйста, убедитесь, что у вас установлена библиотека DJL и скачан артефакт модели "open-unmix" из хранилища
     * Zoo Model перед запуском этого кода.
     *  метод saveAudioComponents принимает NDArray с компонентами музыки, разбитыми на вокал, бас, ударные и другие,
     *  и сохраняет каждый компонент в отдельный аудиофайл формата WAV в указанной директории outputPath.
     *  Компоненты музыки делятся на 4 части с помощью components.split(4, 1).
     * Метод saveAudioFile используется для сохранения компонента в файл. Каждый компонент сохраняется с названием
     * в зависимости от его типа (например, vocals.wav, drums.wav, bass.wav и другие).
     */
}
