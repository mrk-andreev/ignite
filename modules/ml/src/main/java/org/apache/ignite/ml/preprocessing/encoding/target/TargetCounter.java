package org.apache.ignite.ml.preprocessing.encoding.target;

import java.util.HashMap;
import java.util.Map;

/**
 * Counter for encode category.
 */
public class TargetCounter {
  /** */
  private Double targetSum = 0d;

  /** */
  private Long targetCount = 0L;

  /** */
  private Map<String, Long> categoryCounts = new HashMap<>();

  /** */
  private Map<String, Double> categoryTargetSum = new HashMap<>();

  /** */
  public Double getTargetSum() {
    return targetSum;
  }

  /** */
  public void setTargetSum(Double targetSum) {
    this.targetSum = targetSum;
  }

  /** */
  public Long getTargetCount() {
    return targetCount;
  }

  /** */
  public void setTargetCount(Long targetCount) {
    this.targetCount = targetCount;
  }

  /** */
  public Map<String, Long> getCategoryCounts() {
    return categoryCounts;
  }

  /** */
  public void setCategoryCounts(Map<String, Long> categoryCounts) {
    this.categoryCounts = categoryCounts;
  }

  /** */
  public Map<String, Double> getCategoryTargetSum() {
    return categoryTargetSum;
  }

  /** */
  public void setCategoryTargetSum(Map<String, Double> categoryTargetSum) {
    this.categoryTargetSum = categoryTargetSum;
  }
}
