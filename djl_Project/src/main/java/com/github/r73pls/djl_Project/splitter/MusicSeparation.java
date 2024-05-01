package com.github.r73pls.djl_Project.splitter;

public class MusicSeparation {

    public static void main(String[] args) {

        File audioFile = new File("input_audio.wav");

        TarsosDSPAudioFormat format = new TarsosDSPAudioFormat(44100, 16, 1, true, false);
        TarsosDSPAudioInputStream inputStream = new AndroidAudioInputStream(audioFile, format);
        AndroidAudioDispatcher dispatcher = new AndroidAudioDispatcher(inputStream, 1024, 0);

        PitchDetectionHandler handler = new PitchDetectionHandler() {
            @Override
            public void handlePitch(PitchDetectionResult result, AudioEvent e) {
                // Обработка данных о частоте звучащего звука
            }
        };

        AndroidFFMPEGLocator androidFFMPEGLocator = new AndroidFFMPEGLocator();
        dispatcher.addAudioProcessor(androidFFMPEGLocator);

        dispatcher.addAudioProcessor(new AndroidAudioPlayer(format));
        dispatcher.addAudioProcessor(handler);

        dispatcher.run();
    }
    /**
     * В данном примере используется PitchDetectionHandler для обработки данных о частоте звучащего звука.
     * Вы можете дополнить данный код другими обработчиками и процессорами для разделения на различные компоненты
     * музыки. Пожалуйста, убедитесь, что у вас установлены все необходимые зависимости и библиотеки для работы с
     * TarsosDSP.
     */
}

