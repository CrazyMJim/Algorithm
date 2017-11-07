package tinsky

import breeze.linalg.{Axis, DenseMatrix, DenseVector}

class LinearRegression {
  private var theta: DenseVector[Double] = _ //参数向量
  private var historySum: Double = 0.0 //历史损失值
  private var alpha: Double = 0.01 //学习步长
  private var m: Int = 0 //数据量
  private var n: Int = 0 //维度

  def predictY(x: DenseVector[Double]):Unit={
    val appendMatrix: DenseMatrix[Double] = DenseMatrix.ones[Double](1, 1)
    val realX = DenseMatrix.vertcat(appendMatrix,x.toDenseMatrix.t)
    val y = theta.t * realX
    println("Predict Y: "+y)
  }

  def train(trainData: DenseMatrix[Double]): Unit={
    val cols: Int = trainData.cols
    val y = trainData(::,cols - 1)
    val x = trainData.delete(cols - 1, Axis._1)
    train(x,y)
  }

  def train(x: DenseMatrix[Double], y: DenseVector[Double]): Unit={
    m = y.toDenseMatrix.cols
    n = x.cols
    val appendMatrix: DenseMatrix[Double] = DenseMatrix.ones[Double](1, m)
    val realX = DenseMatrix.vertcat(appendMatrix,x.t)
    theta = DenseVector.ones[Double](n + 1)
    var flag: Boolean = true
    var costSum: Double = 0.0

    while (flag){
      costSum = computeCost(realX,y)
      if (historySum == 0.0 || historySum > costSum){
        historySum = costSum
        gradientDescent(realX,y)
      }else if (historySum < costSum){
        alpha = alpha / 10.0
        gradientDescent(realX,y)
      }else if(historySum  == costSum){
        println("********** Finish Train ***********")
        flag = false
      }
    }

    println("theta:" + theta)
  }

  private def computeCost(x: DenseMatrix[Double],y: DenseVector[Double]): Double={
    val predictY = theta.toDenseMatrix * x
    val predictDeviation = predictY.toDenseMatrix.:-(y.toDenseMatrix)
    val result = predictDeviation * predictDeviation.t
    result.toArray(0)
  }

  private def gradientDescent(x: DenseMatrix[Double],y: DenseVector[Double]): Unit={
    val predictY = theta.toDenseMatrix * x
    val predictDeviation = predictY.toDenseMatrix.:-(y.toDenseMatrix)
    val thetaDeviation = predictDeviation * x.t
    theta = theta.:-(thetaDeviation.toDenseVector * alpha / m.toDouble)
  }

}

object LinearRegression{
  def main(args: Array[String]): Unit = {
    val m1: DenseMatrix[Double] = DenseMatrix(
      (1.0,1.0,4.0),
      (2.0,2.0,7.0),
      (3.0,3.0,10.0)
    )

    val m2: DenseMatrix[Double] = DenseMatrix(
      (1.0,4.0),
      (2.0,7.0),
      (3.0,10.0)
    )

    val test1: DenseVector[Double] = DenseVector(5.0,5.0)
    val test2: DenseVector[Double] = DenseVector(5.0)

    val model1 = new LinearRegression()
    println("*************** 一维预测开始 *****************")
    model1.train(m2)
    model1.predictY(test2)
    println("*************** 一维预测结束 *****************")
    println("*************** 二维预测开始 *****************")
    val model = new LinearRegression()
    model.train(m1)
    model.predictY(test1)
    println("*************** 二维预测结束 *****************")
  }



}