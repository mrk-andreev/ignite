package org.apache.ignite.ml.math.distances;

import static org.junit.Assert.assertEquals;


import java.util.Arrays;
import java.util.Collection;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.primitives.vector.impl.DenseVector;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class JensenShannonDistanceTest {
  /** Precision. */
  private static final double PRECISION = 0.01;

  @Parameters(name = "{0}")
  public static Collection<TestData> data() {
    return Arrays.asList(
        new TestData(
            new double[] {1.0, 0.0, 0.0},
            new double[] {0.0, 1.0, 0.0},
            2.0,
            1.0
        ),
        new TestData(
            new double[] {1.0, 0.0},
            new double[] {0.5, 0.5},
            null,
            0.46
        ),
        new TestData(
            new double[] {1.0, 0.0, 0.0},
            new double[] {1.0, 0.5, 0.0},
            null,
            0.36
        )
    );
  }

  private final TestData testData;

  public JensenShannonDistanceTest(TestData testData) {
    this.testData = testData;
  }

  @Test
  public void test() {
    DistanceMeasure distanceMeasure = new JensenShannonDistance(testData.base);

    assertEquals(testData.expRes,
        distanceMeasure.compute(testData.vectorA, testData.vectorB), PRECISION);
    assertEquals(testData.expRes,
        distanceMeasure.compute(testData.vectorA, testData.vectorB), PRECISION);
  }

  private static class TestData {
    public final Vector vectorA;
    public final Vector vectorB;
    public final Double expRes;
    public final Double base;

    private TestData(double[] vectorA, double[] vectorB, Double base, Double expRes) {
      this.vectorA = new DenseVector(vectorA);
      this.vectorB = new DenseVector(vectorB);
      this.base = base;
      this.expRes = expRes;
    }

    @Override
    public String toString() {
      return String.format("d(%s,%s;%s) = %s",
          Arrays.toString(vectorA.asArray()),
          Arrays.toString(vectorB.asArray()),
          base,
          expRes
      );
    }
  }
}
