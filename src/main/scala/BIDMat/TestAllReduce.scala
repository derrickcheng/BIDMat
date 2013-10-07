

import edu.berkeley.bid.comm.{IVec,LVec,Vec}
import edu.berkeley.bid.comm.AllReduce
import scala.actors.Actor._
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.util.concurrent.CountDownLatch

import BIDMat.{Mat, SparseMat, IMat, FMat, SMat}


object TestAllReduce {
  
  def main(args:Array[String]) = {
    val n = args(0).toInt
    val dens = args(1).toFloat
    val ks = args(2).split(",")
    val allks = IMat(ks.length, 1)
    for (i <- 0 until ks.length) allks(i) = ks(i).toInt
    test1(n, dens, allks)
  }
  
  def test1(n:Int, dens:Float, allks:IMat) {
    val a = sprand(n, n, dens)
    val dd = spdiag(n)
    val ad = a + dd
    val b = rand(n, 1)
    //makeSim(ad, b, allks)
    makeTest(ad, b, allks)
  }
  
  def makeSim(a:SMat, b:FMat, allks:IMat) = {
    val M = allks.data.reduceLeft(_*_)
    val bufsize = 5 * (a.nnz / M)
    val ioff = Mat.ioneBased
    val network = new AllReduce(M)
    val rowvecs = new Array[IMat](M)
    val colvecs = new Array[IMat](M)
    val vvecs = new Array[FMat](M)
    val irows = new Array[IVec](M)
    val icols = new Array[IVec](M)
    val smats = new Array[SMat](M)
    
    var model = new Vec(a.nrows)
    var res = new Array[Vec](M)
    
    val (ii, jj, vv) = find3(a)
    val rr = IMat(rand(ii.nrows, 1)*M)
    val counts = accum(rr, 1, M)
    for (i <- 0 until M) {
      rowvecs(i) = IMat(counts(i), 1)
      colvecs(i) = IMat(counts(i), 1)
      vvecs(i) = FMat(counts(i), 1)
      network.simNetwork(i) = new network.Machine(a.nrows, allks.data, i, M, bufsize, true, 2)
    }
    var i = 0
    counts.clear
    while (i < rr.length) {
      val iix = rr(i)
      val ic = counts(iix)
      rowvecs(iix)(ic) = ii(i)
      colvecs(iix)(ic) = jj(i)
      vvecs(iix)(ic) = vv(i)
      counts(iix) = ic+1
      i += 1
    }
    for (i <- 0 until M) {
      val s = new SMat(a.nrows, a.ncols, rowvecs(i).length, SparseMat.incInds(rowvecs(i).data),
          SparseMat.compressInds(colvecs(i).data, a.ncols, new Array[Int](a.ncols+1), rowvecs(i).length), vvecs(i).data)
      irows(i) = new IVec(find(sum(s,2)).data)
      icols(i) = new IVec(find(sum(s,1)).data)
    }
    
    println("compute model")
    for(i <- 0 until M){
       for(j <-0 until irows(i).size){
          model.data(irows(i).data(j)) += vvecs(i).data(j)
       }
    }
    
    val latch = new CountDownLatch(M)
    for (i <- 0 until M) {
      actor {
        network.simNetwork(i).config(irows(i), icols(i))
        res(i) = network.simNetwork(i).reduce(new Vec(vvecs(i).data))
        latch.countDown()
      }
    }
    latch.await();
    println("All done")
    
    println("check model")
    var eps = 0.000001f;
    var maxerr = 0f;
    for(i <- 0 until M){
       for(j <-0 until icols(i).size){
          var err = scala.math.abs( (model.data(icols(i).data(j))-res(i).data(j))/model.data(icols(i).data(j)) )
          if(err > maxerr) { maxerr = err }
          if(err > eps){
            println("machine %d: incorrect, %f, %f".format(i, model.data(icols(i).data(j)), res(i).data(j)))
          }
       }
    }
    println("max error: %f".format(maxerr))
  }
  
  def makeTest(a:SMat, b:FMat, allks:IMat) = {
    val M = allks.data.reduceLeft(_*_)
    val bufsize = 5 * (a.nnz / M)
    val ioff = Mat.ioneBased
    val network = new AllReduce(1)
    val rowvecs = new Array[IMat](M)
    val colvecs = new Array[IMat](M)
    val vvecs = new Array[FMat](M)
    val irows = new Array[IVec](M)
    val icols = new Array[IVec](M)
    val smats = new Array[SMat](M)
    
    var model = new Vec(a.nrows)
 
    
    val (ii, jj, vv) = find3(a)
    val rr = IMat(rand(ii.nrows, 1)*M)
    val counts = accum(rr, 1, M)
    for (i <- 0 until M) {
      rowvecs(i) = IMat(counts(i), 1)
      colvecs(i) = IMat(counts(i), 1)
      vvecs(i) = FMat(counts(i), 1)
    }
    var i = 0
    counts.clear
    while (i < rr.length) {
      val iix = rr(i)
      val ic = counts(iix)
      rowvecs(iix)(ic) = ii(i)
      colvecs(iix)(ic) = jj(i)
      vvecs(iix)(ic) = vv(i)
      counts(iix) = ic+1
      i += 1
    }
    for (i <- 0 until M) {
      val s = new SMat(a.nrows, a.ncols, rowvecs(i).length, SparseMat.incInds(rowvecs(i).data),
          SparseMat.compressInds(colvecs(i).data, a.ncols, new Array[Int](a.ncols+1), rowvecs(i).length), vvecs(i).data)
      irows(i) = new IVec(find(sum(s,2)).data)
      icols(i) = new IVec(find(sum(s,1)).data)
    }
    
    println("compute model")
    for(i <- 0 until M){
       for(j <-0 until irows(i).size){
          model.data(irows(i).data(j)) += vvecs(i).data(j)
       }
    }
    
    val machine = new network.Machine(a.nrows, allks.data, -1, M, bufsize, false, 0)
    val imachine = machine.geti()
    if(imachine == 0){println("start config")}
    machine.config(irows(imachine), icols(imachine))
    if(imachine == 0){println("start reduce")}
    val res = machine.reduce(new Vec(vvecs(imachine).data))
    if(imachine == 0){println("All done")}
    machine.stop()
    
    if(imachine == 0){println("check model")}
    var eps = 0.000001f;
    var maxerr = 0f;
    
    for(j <-0 until icols(imachine).size){
       var err = scala.math.abs( (model.data(icols(imachine).data(j))-res.data(j))/model.data(icols(imachine).data(j)) )
       if(err > maxerr) { maxerr = err }
       if(err > eps){
         println("machine %d: incorrect, %f, %f".format(imachine, model.data(icols(imachine).data(j)), res.data(j)))
       }
    }
   
    println("machine %d: max error: %f".format(imachine, maxerr))
    
  }
  
  
}


