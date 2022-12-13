package mkl;


import tech.v3.datatype.Buffer;
import tech.v3.datatype.DoubleReader;
import ham_fisted.Transformables;
import clojure.lang.IFn;

public class Complex {
  public static Buffer realToComplex(double[] real, Buffer complex) {
    final int rsize = real.length;
    complex.fillRange(0, new DoubleReader() {
	public long lsize() { return rsize * 2; }
	public double readDouble(long idx) {
	  if((idx%2) == 0)
	    return real[(int)idx/2];
	  return 0.0;
	}
	public Object reduce(IFn rfn, Object init) {
	  final IFn.ODO rf = Transformables.toDoubleReductionFn(rfn);
	  for(int idx = 0; idx < rsize; ++idx) {
	    init = rf.invokePrim(init, real[idx]);
	    init = rf.invokePrim(init, 0.0);
	  }
	  return init;
	}
      });
    return complex;
  }
}
