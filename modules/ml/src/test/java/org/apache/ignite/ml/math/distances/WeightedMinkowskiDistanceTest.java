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
public class WeightedMinkowskiDistanceTest {
  /** Precision. */
  private static final double PRECISION = 0.01;

  @Parameters(name = "{0}")
  public static Collection<TestData> data() {
    return Arrays.asList(
        new TestData(
            new double[] {1.0, 0.0, 0.0},
            new double[] {0.0, 1.0, 0.0},
            1,
            new double[] {1, 1, 1},
            2.0
        ),
        new TestData(
            new double[] {1.0, 0.0, 0.0},
            new double[] {0.0, 1.0, 0.0},
            2,
            new double[] {1, 1, 1},
            1.41
        ),
        new TestData(
            new double[] {1.0, 0.0, 0.0},
            new double[] {0.0, 1.0, 0.0},
            3,
            new double[] {1, 1, 1},
            1.25
        )
    );
  }

  private final TestData testData;

  public WeightedMinkowskiDistanceTest(TestData testData) {
    this.testData = testData;
  }

  @Test
  public void test() {
    DistanceMeasure distanceMeasure = new WeightedMinkowskiDistance(testData.p, testData.weight);

    assertEquals(testData.expRes,
        distanceMeasure.compute(testData.vectorA, testData.vectorB), PRECISION);
    assertEquals(testData.expRes,
        distanceMeasure.compute(testData.vectorA, testData.vectorB), PRECISION);
  }

  private static class TestData {
    public final Vector vectorA;
    public final Vector vectorB;
    public final Integer p;
    public final Vector weight;
    public final Double expRes;

    private TestData(double[] vectorA, double[] vectorB, Integer p, double[] weight, double expRes) {
      this.vectorA = new DenseVector(vectorA);
      this.vectorB = new DenseVector(vectorB);
      this.p = p;
      this.weight = new DenseVector(weight);
      this.expRes = expRes;
    }

    @Override
    public String toString() {
      return String.format("d(%s,%s;%s,%s) = %s",
          Arrays.toString(vectorA.asArray()),
          Arrays.toString(vectorB.asArray()),
          p,
          Arrays.toString(weight.asArray()),
          expRes
      );
    }
  }
}
