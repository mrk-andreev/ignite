package org.apache.ignite.ml.math.distances;

import org.apache.ignite.ml.math.exceptions.math.CardinalityException;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.util.MatrixUtil;

/**
 * Calculates the Canberra distance between two points.
 */
public class CanberraDistance implements DistanceMeasure {
  /**
   * Serializable version identifier.
   */
  private static final long serialVersionUID = 1771556549784040092L;

  /**
   * {@inheritDoc}
   */
  @Override
  public double compute(Vector a, Vector b)
      throws CardinalityException {
    Vector top = MatrixUtil.localCopyOf(a).minus(b).map(Math::abs);
    Vector down = MatrixUtil.localCopyOf(a).map(Math::abs)
        .plus(MatrixUtil.localCopyOf(b).map(Math::abs))
        .map(value -> value != 0 ? 1 / value : 0);

    return top.times(down).sum();
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
