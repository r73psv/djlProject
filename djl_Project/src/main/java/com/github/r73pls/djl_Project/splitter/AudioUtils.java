package com.github.r73pls.djl_Project.splitter;

import ai.djl ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.modality.audio.AudioUtils;
import ai.djl.translate.TranslatorContext;
import java.io.File;

public class AudioUtils {

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

}
