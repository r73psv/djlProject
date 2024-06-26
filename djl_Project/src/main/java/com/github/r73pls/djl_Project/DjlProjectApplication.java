package com.github.r73pls.djl_Project;


import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.metric.Metric;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import com.github.r73pls.djl_Project.deepLearning.LeNet;
import com.github.r73pls.djl_Project.imageClassificftion.*;
import com.github.r73pls.djl_Project.mlp.MlpTrainer;
import com.github.r73pls.djl_Project.mlp.TrainModel;
import com.github.r73pls.djl_Project.ndarray.LinearRegressionTrainer;
import com.github.r73pls.djl_Project.ndarray.NdArrayLes1;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import ai.djl.translate.TranslateException;
import java.io.IOException;

@SpringBootApplication
public class DjlProjectApplication {

	public DjlProjectApplication()  {

	}

	public static void main(String[] args) {
		SpringApplication.run(DjlProjectApplication.class, args);
//		NDArray nd=  NdArrayLes1.createNdArray(12);
//		Shape shape = NdArrayLes1.getNDarrayShape(nd);
//		NDArray nd2 = NdArrayLes1.reshapeNDarray(nd,new Shape(2,3,-1));
//		Long sizeArray = NdArrayLes1.getNdArraySize(nd2);
//		NDArray nd3=NdArrayLes1.createNdarray(new Shape(5,6,2));
//		NDArray nd4=NdArrayLes1.createInitializedNDarray(2,new Shape(4,3,2));
//		System.out.println(nd);
//		System.out.println(shape);
//		System.out.println(sizeArray);
//		System.out.println(nd2);
//		System.out.println(nd3);
//		System.out.println(nd4);
//		float[] x = new float[]{1f,2f,3f,4f,5f};
//		float[] y = new float[]{5f,4f,3f,2f,1f};
//		//арифметические операции
//		System.out.println(NdArrayLes1.calcNdarrayFloat(x,y,"+"));
//		System.out.println(NdArrayLes1.calcNdarrayFloat(x,y,"-"));
//		System.out.println(NdArrayLes1.calcNdarrayFloat(x,y,"*"));
//		System.out.println(NdArrayLes1.calcNdarrayFloat(x,y,"/"));
//		System.out.println(NdArrayLes1.calcNdarrayFloat(x,y,"**"));
//		//слияние массивов
//		NDArray nd5=NdArrayLes1.createNdArray(12).reshape(new Shape(3,4));
//		NDArray nd6=NdArrayLes1.createNdArray(12).reshape(new Shape(3,4));
//		System.out.println(NdArrayLes1.concatNDarray(nd5,nd6));
//		System.out.println(nd5.concat(nd6,1));
//		System.out.println(NdArrayLes1.eqNDArray(nd5,nd6));
//		//суммирование элементов массива
//		System.out.println(NdArrayLes1.sumElementNDArray(nd5));
//		//получение части массива
//		System.out.println(NdArrayLes1.getIndex(nd5,"1:5"));
//        //скалярное произведение
//        NDManager manager=NDManager.newBaseManager();
//        NDArray a = manager.create(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1});
//        System.out.println(NdArrayLes1.dotNDArray(a,a));
//		LinearRegressionTrainer.trainModel(10,3);
//		LinearRegressionTrainer.testModel();
//		LinearRegressionTrainer.saveModel(3);

		int batchSize = 256;
		boolean randomShuffle = true;

// get training and validation dataset
		FashionMnist trainingSet = FashionMnist.builder()
				.optUsage(Dataset.Usage.TRAIN)
				.setSampling(batchSize, randomShuffle)
				.optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
				.build();

		FashionMnist validationSet = FashionMnist.builder()
				.optUsage(Dataset.Usage.TEST)
				.setSampling(batchSize, false)
				.optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
				.build();

		int numEpochs = 10;
		float lr = 0.1f;

//		try {
//			TrainingModelSm.trainCh3(Net::net, trainingSet, validationSet, LossFunction::crossEntropy, numEpochs, Updater::updater);
//		} catch (TranslateException e) {
//			throw new RuntimeException(e);
//		}
////
//		Metrics metrics = new Metrics();
//		Trainer trainer = SoftMaxModel.trainer;
//		trainer.initialize(new Shape(1, 28 * 28));
//		trainer.setMetrics(metrics);
//
//		try {
//			EasyTrain.fit(trainer, numEpochs, trainingSet, validationSet);
//		} catch (TranslateException e) {
//			throw new RuntimeException(e);
//		}
//		var result = trainer.getTrainingResult();
//
//	}

//		try {
//			MlpTrainer.train(Net::net, trainingSet, validationSet, Updater::updater);
//		} catch (TranslateException e) {
//			throw new RuntimeException(e);
//		} catch (IOException e) {
//			throw new RuntimeException(e);
//		}
//		TrainModel.train(trainingSet,validationSet);
//		TrainModel.saveModel();
		try {
			LeNet.trainingChapter6(trainingSet,validationSet, numEpochs, LeNet.train());
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (TranslateException e) {
			throw new RuntimeException(e);
		}
	}
}
