package com.github.r73pls.djl_Project.ndarray;

import ai.djl.ndarray.NDArray;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class DataPoints {
    private NDArray X;
    private NDArray y;
}
