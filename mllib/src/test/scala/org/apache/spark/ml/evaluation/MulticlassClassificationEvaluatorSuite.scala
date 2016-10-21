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

import org.apache.spark.{SparkException, SparkFunSuite}
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTestingUtils}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext

class MulticlassClassificationEvaluatorSuite
  extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  test("params") {
    ParamsSuite.checkParams(new MulticlassClassificationEvaluator)
  }

  test("read/write") {
    val evaluator = new MulticlassClassificationEvaluator()
      .setPredictionCol("myPrediction")
      .setLabelCol("myLabel")
      .setMetricName("accuracy")
    testDefaultReadWrite(evaluator)
  }

  test("should support all NumericType labels and not support other types") {
    MLTestingUtils.checkNumericTypes(new MulticlassClassificationEvaluator, spark)
  }

  test("Multiclass evaluation metrics") {
    /*
     * Confusion matrix for 3-class classification with total 9 instances:
     * |2|1|1| true class0 (4 instances)
     * |1|3|0| true class1 (4 instances)
     * |0|0|1| true class2 (1 instance)
     */
    val delta = 1e-7
    val accuracy = (2.0 + 3.0 + 1.0) / ((2 + 3 + 1) + (1 + 1 + 1))
    val precision0 = 2.0 / (2 + 1)
    val precision1 = 3.0 / (3 + 1)
    val precision2 = 1.0 / (1 + 1)
    val recall0 = 2.0 / (2 + 2)
    val recall1 = 3.0 / (3 + 1)
    val recall2 = 1.0 / 1
    val f1measure0 = 2 * precision0 * recall0 / (precision0 + recall0)
    val f1measure1 = 2 * precision1 * recall1 / (precision1 + recall1)
    val f1measure2 = 2 * precision2 * recall2 / (precision2 + recall2)
    val tpRate0 = 2.0 / (2 + 1 + 1)
    val tpRate1 = 3.0 / (1 + 3)
    val tpRate2 = 1.0 / (1 + 1)
    val fpRate0 = 1.0 / (9 - 4)
    val fpRate1 = 1.0 / (9 - 4)
    val fpRate2 = 1.0 / (9 - 1)

    val df = spark.createDataFrame(
      Seq((0.0, 0.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
        (1.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.0, 0.0))).toDF("prediction", "label")

    val evaluator = new MulticlassClassificationEvaluator()

    assert(evaluator.setMetricName("f1").evaluate(df) ~==
      (4.0 / 9) * f1measure0 + (4.0 / 9) * f1measure1 + (1.0 / 9) * f1measure2 relTol delta)
    assert(evaluator.setMetricName("weightedPrecision").evaluate(df) ~==
      (4.0 / 9) * precision0 + (4.0 / 9) * precision1 + (1.0 / 9) * precision2 relTol delta)
    assert(evaluator.setMetricName("weightedRecall").evaluate(df) ~==
      (4.0 / 9) * recall0 + (4.0 / 9) * recall1 + (1.0 / 9) * recall2 relTol delta)
    assert(evaluator.setMetricName("accuracy").evaluate(df) ~== accuracy relTol delta)

    evaluator.setMetricName("f1")
    assert(evaluator.setLabel(0.0).evaluate(df) ~== f1measure0 relTol delta)
    assert(evaluator.setLabel(1.0).evaluate(df) ~== f1measure1 relTol delta)
    assert(evaluator.setLabel(2.0).evaluate(df) ~== f1measure2 relTol delta)

    evaluator.setMetricName("precision")
    assert(evaluator.setLabel(0.0).evaluate(df) ~== precision0 relTol delta)
    assert(evaluator.setLabel(1.0).evaluate(df) ~== precision1 relTol delta)
    assert(evaluator.setLabel(2.0).evaluate(df) ~== precision2 relTol delta)

    evaluator.setMetricName("recall")
    assert(evaluator.setLabel(0.0).evaluate(df) ~== recall0 relTol delta)
    assert(evaluator.setLabel(1.0).evaluate(df) ~== recall1 relTol delta)
    assert(evaluator.setLabel(2.0).evaluate(df) ~== recall2 relTol delta)

    evaluator.setMetricName("truePositiveRate")
    assert(evaluator.setLabel(0.0).evaluate(df) ~== tpRate0 relTol delta)
    assert(evaluator.setLabel(1.0).evaluate(df) ~== tpRate1 relTol delta)
    assert(evaluator.setLabel(2.0).evaluate(df) ~== tpRate2 relTol delta)

    evaluator.setMetricName("falsePositiveRate")
    assert(evaluator.setLabel(0.0).evaluate(df) ~== fpRate0 relTol delta)
    assert(evaluator.setLabel(1.0).evaluate(df) ~== fpRate1 relTol delta)
    assert(evaluator.setLabel(2.0).evaluate(df) ~== fpRate2 relTol delta)

    Array("precision", "recall", "truePositiveRate", "falsePositiveRate").foreach {
      metricName =>
        val thrown = intercept[SparkException] {
          evaluator.unsetLabel.setMetricName(metricName).evaluate(df)
        }
        assert(thrown.getMessage contains s"Metric ${metricName} must be used with label.")
    }
  }
}
