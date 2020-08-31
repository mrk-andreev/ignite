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

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.apache.ignite.ml.inference.parser.ModelParser;
import org.apache.ignite.ml.math.primitives.vector.NamedVector;

/**
 * H2O MOJO model parser.
 */
public class OnnxMojoModelParser implements ModelParser<NamedVector, Double, OnnxMojoModel> {
    /** */
    private static final long serialVersionUID = -170352744966205716L;

    /** {@inheritDoc} */
    @Override public OnnxMojoModel parse(byte[] mojoBytes) {
        try {
            OrtEnvironment environment = OrtEnvironment.getEnvironment();
            return new OnnxMojoModel(
                environment.createSession(mojoBytes, new OrtSession.SessionOptions()));
        } catch (OrtException e) {
            throw new RuntimeException("Failed to parse MOJO", e);
        }
    }
}
