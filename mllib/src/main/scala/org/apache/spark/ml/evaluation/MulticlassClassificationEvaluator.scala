/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.evaluation

import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.{HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

/**
 * :: Experimental ::
 * Evaluator for multiclass classification, which expects two input columns: prediction and label.
 */
@Since("1.5.0")
@Experimental
class MulticlassClassificationEvaluator @Since("1.5.0") (@Since("1.5.0") override val uid: String)
  extends Evaluator with HasPredictionCol with HasLabelCol with DefaultParamsWritable {

  private var label : Option[Double] = None

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("mcEval"))

  /**
   * param for metric name in evaluation (supports `"f1"` (default), `"weightedPrecision"`,
   * `"weightedRecall"`, `"accuracy"`, `"precision"`, `"recall"`, `"truePositiveRate"`,
   * `"falsePositiveRate"`)
   * @group param
   */
  @Since("1.5.0")
  val metricName: Param[String] = {
    val allowedParams = ParamValidators.inArray(Array("f1", "weightedPrecision",
      "weightedRecall", "accuracy", "precision", "recall", "truePositiveRate",
      "falsePositiveRate"))
    new Param(this, "metricName", "metric name in evaluation " +
      "(f1|weightedPrecision|weightedRecall|accuracy|precision|recall|truePositiveRate" +
      "|falsePositiveRate)", allowedParams)
  }

  /** @group getParam */
  @Since("1.5.0")
  def getMetricName: String = $(metricName)

  /** @group setParam */
  @Since("1.5.0")
  def setMetricName(value: String): this.type = set(metricName, value)

  /** @group setParam */
  @Since("1.5.0")
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  @Since("1.5.0")
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /** @group getParam */
  @Since("2.1.0")
  def getLabel: Double = label.get

  /** @group setParam */
  @Since("2.1.0")
  def setLabel(value: Double): this.type = {
    label = Some(value)
    this
  }

  /** @group setParam */
  @Since("2.1.0")
  def unsetLabel: this.type = {
    label = None
    this
  }

  setDefault(metricName -> "f1")

  @Since("2.0.0")
  override def evaluate(dataset: Dataset[_]): Double = {
    if (Set("precision", "recall", "truePositiveRate", "falsePositiveRate")
      .contains($(metricName))) {
      require(label.nonEmpty, s"Metric ${$(metricName)} must be used with label.")
    }

    val schema = dataset.schema
    SchemaUtils.checkColumnType(schema, $(predictionCol), DoubleType)
    SchemaUtils.checkNumericType(schema, $(labelCol))

    val predictionAndLabels =
      dataset.select(col($(predictionCol)), col($(labelCol)).cast(DoubleType)).rdd.map {
        case Row(prediction: Double, label: Double) => (prediction, label)
      }
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val metric = ($(metricName), label) match {
      // weighted metrics
      case ("f1", None) => metrics.weightedFMeasure
      case ("weightedPrecision", _) => metrics.weightedPrecision
      case ("weightedRecall", _) => metrics.weightedRecall
      case ("accuracy", _) => metrics.accuracy
      // metrics per label
      case ("f1", Some(l)) => metrics.fMeasure(l)
      case ("precision", Some(l)) => metrics.precision(l)
      case ("recall", Some(l)) => metrics.recall(l)
      case ("truePositiveRate", Some(l)) => metrics.truePositiveRate(l)
      case ("falsePositiveRate", Some(l)) => metrics.falsePositiveRate(l)
    }
    metric
  }

  @Since("1.5.0")
  override def isLargerBetter: Boolean = $(metricName) match {
    case "f1" => true
    case "weightedPrecision" => true
    case "weightedRecall" => true
    case "accuracy" => true
    case "precision" => true
    case "recall" => true
    case "truePositiveRate" => true
    case "falsePositiveRate" => false
  }

  @Since("1.5.0")
  override def copy(extra: ParamMap): MulticlassClassificationEvaluator = defaultCopy(extra)
}

@Since("1.6.0")
object MulticlassClassificationEvaluator
  extends DefaultParamsReadable[MulticlassClassificationEvaluator] {

  @Since("1.6.0")
  override def load(path: String): MulticlassClassificationEvaluator = super.load(path)
}
