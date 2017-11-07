package test.group

import breeze.linalg.{Axis, DenseMatrix, DenseVector}

import scala.collection.mutable.ArrayBuffer

class Perceptron{
  private var x: DenseMatrix[Double] = _
  private var y: DenseVector[Double] = _
  private var weight: DenseVector[Double] = _
  private var bias: Double = 0.0
  private var alpha: Double = 1.0
  private var n: Int = 0
  private var m: Int = 0

  def this(trainSet: DenseMatrix[Double]){
    this()
    this.m = trainSet.rows
    this.n = trainSet.cols - 1
    this.y = trainSet(::,n)
    this.x = trainSet.delete(n,Axis._1)
    weight = DenseVector.zeros[Double](n)
  }

  def this(trainSet: DenseMatrix[Double],weight: DenseVector[Double]){
    this(trainSet)
    this.weight = weight
  }

  def train():Unit={
    var stop: Int = 0
    var flag: Int = 0

    while(flag < m){
      val tmpY = weight.toDenseMatrix * x(stop,::).t + bias
      val dotNum: Double = y.valueAt(stop) * tmpY.data(0)
      var tmp:DenseVector[Double] = null

      if(dotNum <= 0){
        tmp = x(stop,::).t
        weight = weight + (tmp * (alpha * y.valueAt(stop)))
        bias = bias + alpha * y.valueAt(stop)
        flag = 0
      }else{
        stop += 1
        flag += 1
      }

      if(stop == m && flag != m)
        stop = 0
    }

  }

  def train(trainSet: DenseMatrix[Double]): Unit={
    this.m = trainSet.rows
    this.n = trainSet.cols - 1
    this.y = trainSet(::,n + 1)
    this.x = trainSet.delete(n,Axis._1)
    train()
  }

  def fit(x: DenseVector[Double]): Int={
    val predictY = weight.toDenseMatrix * x.toDenseMatrix.t + bias
    if( predictY.data(0) > 0)
      1
    else
      -1
  }

  def getWeight(): DenseVector[Double] = this.weight

  //设置初始权重
  def setWeight(w: DenseVector[Double]) = this.weight = w

  def getBias(): Double = this.bias

  //设置初始乖离值
  def setBias(b: Double) = this.bias = b

  def getAlpha(): Double = this.alpha

  def setAlpha(alpha: Double) = this.alpha = alpha

}

object Perceptron{
  def main(args: Array[String]): Unit = {
    val features: DenseMatrix[Double] = DenseMatrix(
      (3.0,3.0,3.0,1.0),
      (4.0,3.0,2.0,1.0),
      (1.0,1.0,1.0,-1.0)
    )

    val test: DenseVector[Double] = DenseVector(4.0,4.0,4.0)

    val perceptron: Perceptron = new Perceptron(features)
    perceptron.train()
    val label: Int = perceptron.fit(test)
    println("*************** Simple Perceptron Check Start ********************")
    println("Check feature:" + test + "  category:" + label)
    println("Perceptron Weight:" + perceptron.getWeight())
    println("Perceptron Bias:" + perceptron.getBias())
    println("*************** Simple Perceptron Check End ********************")

  }
}