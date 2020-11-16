package org.apache.ignite.ml.preprocessing.encoding.target;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Metadata for encode category.
 */
public class TargetEncodingMeta {
  /** */
  private final Double globalMean;

  /** */
  private final Map<String, Double> categoryMean;

  /** */
  public TargetEncodingMeta(Double globalMean,
                            Map<String, Double> categoryMean) {
    this.globalMean = globalMean;
    this.categoryMean = new HashMap<>(categoryMean);
  }

  /** */
  public Double getGlobalMean() {
    return globalMean;
  }

  /** */
  public Map<String, Double> getCategoryMean() {
    return Collections.unmodifiableMap(categoryMean);
  }
}
