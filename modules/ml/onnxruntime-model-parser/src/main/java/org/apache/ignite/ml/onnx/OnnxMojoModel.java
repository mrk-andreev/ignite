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

package org.apache.ignite.ml.onnx;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.apache.ignite.ml.inference.Model;
import org.apache.ignite.ml.math.primitives.vector.NamedVector;

/**
 * H2O MOJO Model imported and wrapped to be compatible with Apache Ignite infrastructure.
 */
public class OnnxMojoModel implements Model<NamedVector, Double> {
    /** H2O MOJO model (accessible using in EasyPredict API). */
    private final OrtSession model;

    /**
     * Constructs a new instance of H2O MOJO model wrapper.
     *
     * @param model H2O MOJO Model
     */
    public OnnxMojoModel(OrtSession model) {
        this.model = model;
    }

    /** {@inheritDoc} */
    @Override public Double predict(NamedVector input) {
//        model.run()
        return 0d;
    }

    /** {@inheritDoc} */
    @Override public void close() {
        try {
            model.close();
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }
}
