package org.apache.ignite.ml.math.distances;


import org.apache.ignite.ml.math.exceptions.math.CardinalityException;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.util.MatrixUtil;

/**
 * Calculates the Weighted Minkowski distance between two points.
 */
public class WeightedMinkowskiDistance implements DistanceMeasure {
  /**
   * Serializable version identifier.
   */
  private static final long serialVersionUID = 1771556549784040096L;

  private final int p;

  private final Vector weight;

  public WeightedMinkowskiDistance(int p, Vector weight) {
    this.p = p;
    this.weight = weight.copy().map(x -> Math.pow(x, p)).map(x -> Math.pow(x, 1 / (double) p));
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public double compute(Vector a, Vector b)
      throws CardinalityException {

    return Math.pow(
        MatrixUtil.localCopyOf(a).minus(b)
            .map(x -> Math.pow(Math.abs(x), p))
            .times(weight)
            .sum(),
        1 / (double) p
    );
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
