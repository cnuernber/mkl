package mkl;


import com.sun.jna.Library;
import com.sun.jna.Pointer;
import ham_fisted.IMutList;


public interface DFTI extends Library {
  public long DftiCreateDescriptor(Pointer ptrToHandle, int precision, int domain,
				   long dimension, Object... args);
  public long DftiCommitDescriptor(Pointer handle);
  public long DftiSetValue(Pointer handle, int config_param, Object... args);
  public long DftiGetValue(Pointer handle, int config_param, Object... args);
  public long DftiComputeForward(Pointer handle, Pointer data, Object... args);
  public long DftiComputeBackward(Pointer handle, Pointer data, Object... args);
  public long DftiFreeDescriptor(Pointer ptrToHandle);
  public Pointer DftiErrorMessage(long error);
}
