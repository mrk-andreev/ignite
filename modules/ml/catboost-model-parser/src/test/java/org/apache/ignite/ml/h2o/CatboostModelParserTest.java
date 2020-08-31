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

package org.apache.ignite.ml.h2o;

import ai.catboost.CatBoostModel;
import java.net.URL;
import java.util.HashMap;

import org.apache.ignite.ml.catboost.CatboostModelParser;
import org.apache.ignite.ml.catboost.CatboostMojoModel;
import org.apache.ignite.ml.inference.builder.SingleModelBuilder;
import org.apache.ignite.ml.inference.builder.SyncModelBuilder;
import org.apache.ignite.ml.inference.reader.FileSystemModelReader;
import org.apache.ignite.ml.inference.reader.ModelReader;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Tests for {@link CatboostModelParser}.
 */
public class CatboostModelParserTest {
    /** Test model resource name. */
    private static final String TEST_MODEL_RESOURCE = "mojos/model.cbm";

    /** Parser. */
    private final CatboostModelParser parser = new CatboostModelParser();

    /** Model builder. */
    private final SyncModelBuilder mdlBuilder = new SingleModelBuilder();

    /** */
    @Test
    public void testParseAndPredict() {
        URL url = CatboostModelParserTest.class.getClassLoader().getResource(TEST_MODEL_RESOURCE);
        if (url == null)
            throw new IllegalStateException("File not found [resource_name=" + TEST_MODEL_RESOURCE + "]");

        ModelReader reader = new FileSystemModelReader(url.getPath());

        try (CatboostMojoModel mdl = mdlBuilder.build(reader, parser)) {
            HashMap<String, Double> input = new HashMap<>();
            input.put("Pclass", 3.0);
            input.put("Sex", 0.0);
            input.put("Embarked", 3.0);
            input.put("Fare", 7.9250);
            input.put("Age", 26.0);

            double prediction = mdl.predict(VectorUtils.of(input));

            assertEquals(0.90, prediction, 1e-5);
        }
    }
}
