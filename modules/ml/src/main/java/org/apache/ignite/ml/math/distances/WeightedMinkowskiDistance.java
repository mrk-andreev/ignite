package org.apache.ignite.ml.math.distances;

import static org.apache.ignite.ml.math.functions.Functions.PLUS;


import org.apache.ignite.ml.math.exceptions.math.CardinalityException;
import org.apache.ignite.ml.math.functions.IgniteDoubleFunction;
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

  private final double p;

  private final Vector weights;

  public WeightedMinkowskiDistance(double p, Vector weights) {
    this.p = p;
    this.weights = weights.copy().map(x -> Math.pow(x, p)).map(x -> Math.pow(x, 1 / p));
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public double compute(Vector a, Vector b)
      throws CardinalityException {

    return MatrixUtil.localCopyOf(a).minus(b).times(weights).kNorm(p);
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
