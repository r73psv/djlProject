package com.github.r73pls.djl_Project;


import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import com.github.r73pls.djl_Project.ndarray.NdArrayLes1;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;

@SpringBootApplication
public class DjlProjectApplication {

	public DjlProjectApplication()  {

	}

	public static void main(String[] args) throws IOException {
		SpringApplication.run(DjlProjectApplication.class, args);
		NDArray nd=  NdArrayLes1.createNdArray(12);
		Shape shape = NdArrayLes1.getNDarrayShape(nd);
		NDArray nd2 = NdArrayLes1.reshapeNDarray(nd,new Shape(2,3,-1));
		Long sizeArray = NdArrayLes1.getNdArraySize(nd2);
		NDArray nd3=NdArrayLes1.createNdarray(new Shape(5,6,2));
		NDArray nd4=NdArrayLes1.createInitializedNDarray(2,new Shape(4,3,2));
		System.out.println(nd);
		System.out.println(shape);
		System.out.println(sizeArray);
		System.out.println(nd2);
		System.out.println(nd3);
		System.out.println(nd4);
		float[] x = new float[]{1f,2f,3f,4f,5f};
		float[] y = new float[]{5f,4f,3f,2f,1f};
		//арифметические операции
		System.out.println(NdArrayLes1.calcNdarrayFloat(x,y,"+"));
		System.out.println(NdArrayLes1.calcNdarrayFloat(x,y,"-"));
		System.out.println(NdArrayLes1.calcNdarrayFloat(x,y,"*"));
		System.out.println(NdArrayLes1.calcNdarrayFloat(x,y,"/"));
		System.out.println(NdArrayLes1.calcNdarrayFloat(x,y,"**"));
		//слияние массивов
		NDArray nd5=NdArrayLes1.createNdArray(12).reshape(new Shape(3,4));
		NDArray nd6=NdArrayLes1.createNdArray(12).reshape(new Shape(3,4));
		System.out.println(NdArrayLes1.concatNDarray(nd5,nd6));
		System.out.println(nd5.concat(nd6,1));
		System.out.println(NdArrayLes1.eqNDArray(nd5,nd6));
		//суммирование элементов массива
		System.out.println(NdArrayLes1.sumElementNDArray(nd5));
		//получение части массива
		System.out.println(NdArrayLes1.getIndex(nd5,"1:5"));
        //скалярное произведение
        NDManager manager=NDManager.newBaseManager();
        NDArray a = manager.create(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1});
        System.out.println(NdArrayLes1.dotNDArray(a,a));

	}




}
