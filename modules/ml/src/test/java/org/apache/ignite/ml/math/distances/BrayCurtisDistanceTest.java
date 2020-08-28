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
public class BrayCurtisDistanceTest {
  /** Precision. */
  private static final double PRECISION = 0.01;

  @Parameters(name = "{0}")
  public static Collection<TestData> data() {
    return Arrays.asList(
        new TestData(
            new double[] {0, 0, 0},
            new double[] {2, 1, 0},
            1.0
        ),
        new TestData(
            new double[] {1, 2, 3},
            new double[] {2, 1, 0},
            0.55
        ),
        new TestData(
            new double[] {1, 2, 3},
            new double[] {2, 1, 50},
            0.83
        ),
        new TestData(
            new double[] {1, -100, 3},
            new double[] {2, 1, -50},
            1.04
        )
    );
  }

  private final TestData testData;

  public BrayCurtisDistanceTest(TestData testData) {
    this.testData = testData;
  }

  @Test
  public void test() {
    DistanceMeasure distanceMeasure = new BrayCurtisDistance();

    assertEquals(testData.expRes,
        distanceMeasure.compute(testData.vectorA, testData.vectorB), PRECISION);
    assertEquals(testData.expRes,
        distanceMeasure.compute(testData.vectorA, testData.vectorB), PRECISION);
  }

  private static class TestData {
    public final Vector vectorA;
    public final Vector vectorB;
    public final double expRes;

    private TestData(double[] vectorA, double[] vectorB, double expRes) {
      this.vectorA = new DenseVector(vectorA);
      this.vectorB = new DenseVector(vectorB);
      this.expRes = expRes;
    }

    @Override
    public String toString() {
      return String.format("d(%s,%s) = %s",
          Arrays.toString(vectorA.asArray()),
          Arrays.toString(vectorB.asArray()),
          expRes
      );
    }
  }
}
