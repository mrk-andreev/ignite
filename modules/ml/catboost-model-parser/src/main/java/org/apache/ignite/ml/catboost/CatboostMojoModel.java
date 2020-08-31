/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.ignite.ml.catboost;

import ai.catboost.CatBoostError;
import ai.catboost.CatBoostModel;
import org.apache.ignite.ml.inference.Model;
import org.apache.ignite.ml.math.primitives.vector.NamedVector;

/**
 * H2O MOJO Model imported and wrapped to be compatible with Apache Ignite infrastructure.
 */
public class CatboostMojoModel implements Model<NamedVector, Double> {
    /** H2O MOJO model (accessible using in EasyPredict API). */
    private final CatBoostModel catBoostModel;

    /**
     * Constructs a new instance of Catboost MOJO model wrapper.
     *
     * @param catBoostModel MOJO Model
     */
    public CatboostMojoModel(CatBoostModel catBoostModel) {
        this.catBoostModel = catBoostModel;
    }

    /** {@inheritDoc} */
    @Override public Double predict(NamedVector input) {
        try {
            return catBoostModel.predict(
                toFeatureVector(input, catBoostModel),
                catBoostModel.getFeatureNames()
            ).get(0, 0);
        } catch (CatBoostError e) {
            throw new RuntimeException("Failed to predict CatBoostModel", e);
        }
    }

    /**
     * Convert double array to float array
     * @param vector
     * @param model
     */
    private static float[] toFeatureVector(NamedVector vector, CatBoostModel model) {
        String[] features = model.getFeatureNames();
        float[] floatValues = new float[vector.size()];
        for (int i=0; i<floatValues.length; i++) {
            floatValues[i] = (float) vector.get(features[i]);
        }
        return floatValues;
    }

    /** {@inheritDoc} */
    @Override public void close() {
        // nothing to do
    }
}
