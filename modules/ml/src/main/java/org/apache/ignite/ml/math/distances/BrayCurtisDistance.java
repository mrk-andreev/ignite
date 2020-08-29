package org.apache.ignite.ml.math.distances;

import org.apache.ignite.ml.math.exceptions.math.CardinalityException;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.util.MatrixUtil;

/**
 * Calculates the Bray Curtis distance between two points.
 */
public class BrayCurtisDistance implements DistanceMeasure {
  /** Serializable version identifier. */
  private static final long serialVersionUID = 1771556549784040091L;

  /** {@inheritDoc} */
  @Override public double compute(Vector a, Vector b)
      throws CardinalityException {
    double diff = MatrixUtil.localCopyOf(a).minus(b).kNorm(1);
    double sum = MatrixUtil.localCopyOf(a).plus(b).kNorm(1);

    return diff / sum;
  }

  /** {@inheritDoc} */
  @Override public boolean equals(Object obj) {
    if (this == obj)
      return true;

    return obj != null && getClass() == obj.getClass();
  }

  /** {@inheritDoc} */
  @Override public int hashCode() {
    return getClass().hashCode();
  }
}
