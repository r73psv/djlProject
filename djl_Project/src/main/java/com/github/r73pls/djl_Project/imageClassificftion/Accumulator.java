package com.github.r73pls.djl_Project.imageClassificftion;

import java.util.Arrays;

public class Accumulator {
          float[] data;

        public Accumulator(int n) {
            data = new float[n];
        }

        /* Adds a set of numbers to the array */
        public void add(float[] args) {
            for (int i = 0; i < args.length; i++) {
                data[i] += args[i];
            }
        }

        /* Resets the array */
        public void reset() {
            Arrays.fill(data, 0f);
        }

        /* Returns the data point at the given index */
        public float get(int index) {
            return data[index];
        }
    }

