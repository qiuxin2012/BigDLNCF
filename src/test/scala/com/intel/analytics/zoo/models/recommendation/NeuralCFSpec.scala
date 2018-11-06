/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.models.recommendation

import java.net.URL

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.zoo.examples.mlperf.recommendation.{HitRate, Ndcg, NeuralCF}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class NeuralCFSpec extends FlatSpec with Matchers with BeforeAndAfter {

  var sqlContext: SQLContext = _
  var sc: SparkContext = _
  var datain: DataFrame = _
  val userCount = 100
  val itemCount = 100

  before {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("NCFTest")
    sqlContext = SQLContext.getOrCreate(sc)
    Engine.init(1, 4, true)
    val resource: URL = getClass.getClassLoader.getResource("recommender")
    datain = sqlContext.read.parquet(resource.getFile)
      .select("userId", "itemId", "label")
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "NeuralCF without MF forward and backward" should "work properly" in {

    val model = NeuralCF[Float](userCount, itemCount, 5, 5, 5, Array(10, 5), false)
    val ran = new Random(42L)
    val data: Seq[Tensor[Float]] = (1 to 50).map(i => {
      val uid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val iid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val feature: Tensor[Float] = Tensor(T(T(uid, iid)))
      println(feature.size().toList)
      val label = Math.abs(ran.nextInt(4)).toFloat + 1
      println(feature.size())
      feature
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

  "NeuralCF with MF forward and backward" should "work properly" in {
    val model = NeuralCF[Float](userCount, itemCount, 5, 5, 5, Array(10, 5), true, 3)
    val ran = new Random(42L)
    val data: Seq[Tensor[Float]] = (1 to 50).map(i => {
      val uid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val iid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val feature: Tensor[Float] = Tensor(T(T(uid, iid)))
      val label = Math.abs(ran.nextInt(4)).toFloat + 1
      feature
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }


  "hitrate@10" should "works fine" in {
    val o = Tensor[Float].range(1, 1000, 1).apply1(_ / 1000)
    val t = Tensor[Float](1000).zero
    t.setValue(1000, 1)
    val hr = new HitRate[Float]()
    val r1 = hr.apply(o, t).result()
    r1._1 should be (1.0)

    o.setValue(1000, 0.9988f)
    val r2 = hr.apply(o, t).result()
    r2._1 should be (1.0)

    o.setValue(1000, 0.9888f)
    val r3 = hr.apply(o, t).result()
    r3._1 should be (0.0f)
  }

  "ndcg" should "works fine" in {
    val o = Tensor[Float].range(1, 1000, 1).apply1(_ / 1000)
    val t = Tensor[Float](1000).zero
    t.setValue(1000, 1)
    val ndcg = new Ndcg[Float]()
    val r1 = ndcg.apply(o, t).result()
    r1._1 should be (1.0)

    o.setValue(1000, 0.9988f)
    val r2 = ndcg.apply(o, t).result()
    r2._1 should be (0.63092977f)

    o.setValue(1000, 0.9888f)
    val r3 = ndcg.apply(o, t).result()
    r3._1 should be (0.0f)
  }

}

