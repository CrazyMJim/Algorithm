package test.group

import breeze.linalg.DenseVector

import scala.collection.mutable.ArrayBuffer

class Perceptron{
  private var vectors: ArrayBuffer[(DenseVector[Double] , Int)] = null
  private var weight: DenseVector[Double] = null
  private var bias: Double = 0.0
  private var step: Double = 1.0

  def this(vectors: ArrayBuffer[(DenseVector[Double] , Int)]){
    this()
    this.vectors = vectors
    weight = DenseVector.zeros[Double](vectors(0)._1.length)
  }

  def this(vectors: ArrayBuffer[(DenseVector[Double] , Int)],weight: DenseVector[Double]){
    this()
    this.vectors = vectors
    this.weight = weight
  }

  def train():Unit={
    var stop: Int = 0
    var flag: Int = 0

    while(flag < vectors.length){
      val dotNum: Double = vectors(stop)._2 * ( (weight dot vectors(stop)._1) + bias )
      var tmp:DenseVector[Double] = null

      if(dotNum <= 0){
        tmp = vectors(stop)._1.copy
        weight = weight + (tmp :*= (step * vectors(stop)._2))
        bias = bias + vectors(stop)._2 * step
        flag = 0
      }else{
        stop += 1
        flag += 1
      }

      if(stop == vectors.length && flag != vectors.length)
        stop = 0
    }

  }

  def train(vetros: ArrayBuffer[(DenseVector[Double],Int)]): Unit={
    this.vectors = vetros
    train()
  }

  def fit(x: DenseVector[Double]): Int={
    if( ( (weight dot x) + bias ) > 0)
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

  def getStep(): Double = this.step

  def setStep(step: Double) = this.step = step

}

object Perceptron{
  def main(args: Array[String]): Unit = {
    val features: ArrayBuffer[(DenseVector[Double] , Int)] = ArrayBuffer(
      ( DenseVector(3.0,3.0,3.0) ,1),
      ( DenseVector(4.0,3.0,2.0) ,1),
      ( DenseVector(1.0,1.0,1.0) ,-1)
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