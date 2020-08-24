package org.apache.ignite.ml.math.distances;

import org.apache.ignite.ml.math.exceptions.math.CardinalityException;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.util.MatrixUtil;

/**
 * Calculates the JensenShannonDistance distance between two points.
 */
public class JensenShannonDistance implements DistanceMeasure {
  /**
   * Serializable version identifier.
   */
  private static final long serialVersionUID = 1771556549784040093L;

  private final Double base;

  public JensenShannonDistance(){
    base = null;
  }

  public JensenShannonDistance(Double base){
    this.base = base;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public double compute(Vector a, Vector b)
      throws CardinalityException {
    Vector aNormalized = MatrixUtil.localCopyOf(a).divide(a.sum());
    Vector bNormalized = MatrixUtil.localCopyOf(b).divide(b.sum());

    Vector mean = aNormalized.plus(bNormalized).divide(2d);

    double js = aNormalized.map(mean, this::relativeEntropy).sum() +
        bNormalized.map(mean, this::relativeEntropy).sum();

    if (base != null) {
      js /= Math.log(base);
    }

    return Math.sqrt(js / 2d);
  }

  private double relativeEntropy(double x, double y) {
    if (x > 0 && y > 0) {
      return x * Math.log(x / y);
    }
    if (x == 0 && y >= 0) {
      return 0;
    }

    return Double.POSITIVE_INFINITY;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }

    return obj != null && getClass() == obj.getClass();
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int hashCode() {
    return getClass().hashCode();
  }
}
