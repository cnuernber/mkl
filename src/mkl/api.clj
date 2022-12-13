(ns mkl.api
  (:require [mkl.ffi :as ffi]
            [tech.v3.datatype :as dt]
            [tech.v3.datatype.ffi :as dt-ffi]
            [tech.v3.datatype.copy :as copy]
            [tech.v3.datatype.casting :as casting]
            [tech.v3.datatype.native-buffer :as nbuf]
            [tech.v3.resource :as resource]
            [tech.v3.tensor :as dtt]
            [tech.v3.resource :as resource]
            [ham-fisted.api :as hamf])
  (:import [org.jtransforms.fft DoubleFFT_1D]
           [tech.v3.datatype Complex]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defn initialize!
  "Initialize - find the `mkl_rt` shared library and load it.  Uses normal OS pathways to
  find the library in addition to the `java_library_path` system variable."
  ([] (ffi/initialize!))
  ([options] (ffi/initialize! options)))


(defn- alloc-uninitialized
  [dtype ^long ec]
  (-> (nbuf/malloc (* ec (casting/numeric-byte-width dtype))
                   {:uninitialized? true})
      (nbuf/set-native-datatype dtype)))


(defn- alloc-zeros
  [dtype ^long ec]
  (-> (nbuf/malloc (* ec (casting/numeric-byte-width dtype)))
      (nbuf/set-native-datatype dtype)))


(defn axpy!
  "Mutable [[axpy]] placing results in rhs."
  [mult lhs rhs]
  (let [dt (dt/elemwise-datatype lhs)
        ec (dt/ecount lhs)
        lstride (long (if (dtt/tensor? lhs)
                        (apply min ((dtt/tensor->dimensions lhs) :strides))
                        1))
        rstride (long (if (dtt/tensor? rhs)
                        (apply min ((dtt/tensor->dimensions rhs) :strides))
                        1))]
    (when-not (identical? dt (dt/elemwise-datatype rhs))
      (throw (Exception. "Datatypes do not match")))
    (when-not (== ec (dt/ecount rhs))
      (throw (Exception. "Element counts do not match")))
    (case dt
      :float32 (ffi/cblas_saxpy ec (float mult) lhs lstride rhs rstride)
      :float64 (ffi/cblas_daxpy ec (double mult) lhs lstride rhs rstride))
    rhs))


(defn axpy
  "Perform a*x+y where a is a constant and x and y native buffers matching in datatype
  and length.  At this time only `:float32` and `:float64` datatypes are supported.

  Meant to be used within a stack resource context.  See docs for [[sub]]."
  [mult lhs rhs]
  (axpy! mult lhs (nbuf/clone-native rhs)))


(defn sub!
  "Subtract two native buffers placing result into `ret`.  See docs for [[sub]]."
  [a b ret]
  (let [aec (dt/ecount a)
        adt (dt/elemwise-datatype a)]
    (when-not (and (== aec (dt/ecount b))
                   (== aec (dt/ecount ret)))
      (throw (Exception. "ecounts do not match.")))
    (when-not (and (identical? adt (dt/elemwise-datatype b))
                   (identical? adt (dt/elemwise-datatype ret)))
      (throw (Exception. "datatypes do not match.")))
    (case adt
      :float32 (ffi/vsSub aec a b ret)
      :float64 (ffi/vdSub aec a b ret)))
  ret)


(defn sub
  "Subtract two native buffers.  No care has been taken to ensure buffers are native or to
  convert them.

  Meant to be used within a stack resource context.

```clojure
mkl.api> (def fdata (dt/make-container :native-heap :float32 (hamf/range 100000)))
#'mkl.api/fdata
mkl.api> (crit/quick-bench (resource/stack-resource-context
                            (sub (hamf/subvec fdata 0 (dec (dt/ecount fdata)))
                                 (hamf/subvec fdata 1))))
Evaluation count : 27468 in 6 samples of 4578 calls.
             Execution time mean : 21.781143 µs
    Execution time std-deviation : 82.384894 ns
   Execution time lower quantile : 21.689956 µs ( 2.5%)
   Execution time upper quantile : 21.872100 µs (97.5%)
                   Overhead used : 1.998471 ns
```"
  [a b]
  (let [aec (dt/ecount a)
        adt (dt/elemwise-datatype a)]
    (sub! a b (alloc-uninitialized adt aec))))


(defn real->complex
  [^doubles real]
  (let [rc (dt/ecount real)
        rvec (alloc-uninitialized :float64 rc)
        _ (copy/unsafe-copy-memory real rvec :float64 rc)
        rval (alloc-zeros :float64 (* 2 rc))]
    (ffi/vdUnpackI rc rvec rval 2)
    rval))


(defn fft-forward-fn
  "Create a function to process forward fft.  Returned function takes
  two args, input and output and returns the output after each invocation.  Input and output
  lengths must match len.  If domain is complex, input and output lengths must be double
  len.

  Complex inputs are stored packed as if in a struct - r1,i1,r2,i2

  * `dtype` - either `:float32` or `:float64`.
  * `domain` - either `:real` or `:complex`."
  [dtype domain ^long len]
  (let [desc (ffi/DftiCreateDescriptor dtype domain len)
        checklen (long (case domain
                         :real len
                         :complex (* 2 len)))]
    (fn [input output]
      (when-not (and (== checklen (dt/ecount input))
                     (== checklen (dt/ecount output)))
        (throw (RuntimeException.
                (format "Input,Output lens are incorrect.  Expected (%s), got %s,%s"
                        checklen (dt/ecount input) (dt/ecount output)))))
      (when-not (and (identical? dtype (dt/elemwise-datatype input))
                     (identical? dtype (dt/elemwise-datatype output)))
        (throw (RuntimeException.
                (format "Input,Output datatypes are incorrect.  Expected (%s), got %s,%is"
                        dtype (dt/elemwise-datatype input) (dt/elemwise-datatype output)))))
      (ffi/DftiComputeForward desc input output))))


(defn correlation1d-fn
  "Create a correlation function that returns data the same length as the input.
  This function is *not* reentrant, do not call simultaneously from multiple threads.

  * `dtype` - One of `:float32` `:float64`.
  * `nsignal` - Length of input signal.
  * `nwin - Number of elements in the kernel.


  Options:

  - `:mode` - One of `:full` `:same`
  - `:algorithm` - One of `:fft` `:naive` `:auto`."
  [dtype nsignal nwin & {:keys [mode algorithm]
                         :or {mode :full
                              algorithm :auto}}]
  (let [nwin (long nwin)
        nsignal (long nsignal)
        full-len (dec (+ nsignal nwin))
        hdl-addr (resource/stack-resource-context
                  (let [hdl-ptr (dt-ffi/make-ptr :pointer 0)
                        mode (long (case algorithm
                                     :fft 2
                                     :naive 1
                                     :auto 0))]
                    (ffi/check-vsl "vslsCorrNewTask1D error: "
                                   (case dtype
                                     :float32 (ffi/vslsCorrNewTask1D hdl-ptr mode
                                                                     nwin full-len full-len)
                                     :float64 (ffi/vsldCorrNewTask1D hdl-ptr mode
                                                                     nwin full-len full-len)))
                    (hdl-ptr 0)))
        hdl (tech.v3.datatype.ffi.Pointer. hdl-addr)
        output (alloc-uninitialized dtype full-len)
        input (alloc-zeros dtype full-len)
        sigbuffer (dt/sub-buffer input 0 nsignal)
        kernel (alloc-uninitialized dtype nwin)
        retval (case mode
                 :same (dt/sub-buffer output (quot nwin 2) nsignal)
                 :full output)]
    (-> (fn [sigdata kdata]
          (dt/copy! kdata kernel)
          (dt/copy! sigdata sigbuffer)
          (ffi/check-vsl
           "Error executing correlation: "
           (case dtype
             :float32 (ffi/vslsCorrExec1D hdl kernel 1 input 1 output 1)
             :float64 (ffi/vsldCorrExec1D hdl kernel 1 input 1 output 1)))
          retval)
        (resource/track {:track-type :auto
                         :dispose-fn #(resource/stack-resource-context
                                       (ffi/check-vsl
                                        "vslCorrDeleteTask: "
                                        (ffi/vslCorrDeleteTask
                                         (dt-ffi/make-ptr :pointer hdl-addr))))}))))

(defn correlate1d
  "Drop in replacement for dtype-next's convolve/correlate1d pathway that is significantly
  faster for large input data and kernel values."
  ([data kernel] (corr data kernel nil))
  ([data kernel options]
   (resource/stack-resource-context
    (dt/->array
     ((correlation1d-fn (get options :datatype :float32)
                        (dt/ecount data) (dt/ecount kernel) options) data kernel)))))


(comment
  (do
    (def small-src (double-array (take 256 (cycle [1 2 3 4 5]))))
    (def large-src (double-array (take 8192 (cycle [1 2 3 4 5]))))
    (def large-large-src (double-array (take (* 8192 2) (cycle [1 2 3 4 5]))))

    (defn jfft-fn
      [^long n]
      (let [fft (DoubleFFT_1D. n)]
        (fn [src-data]
          (let [rval (Complex/realToComplex src-data)]
            (.complexForward fft rval)
            rval))))

    (defn mkl-fft-fn
      [^long n]
      (let [forward-fn (fft-forward-fn :float64 :complex n)
            res (alloc-uninitialized :float64 (* 2 n))]
        (fn [^doubles v]
          (resource/stack-resource-context
           (forward-fn (real->complex v) res)))))

    (def jfft-small (jfft-fn (dt/ecount small-src)))
    (def jfft-large (jfft-fn (dt/ecount large-src)))
    (def jfft-large-large (jfft-fn (dt/ecount large-large-src)))


    (def mkl-fft-small (mkl-fft-fn 256))
    (def mkl-fft-large (mkl-fft-fn 8192))
    (def mkl-fft-large-large (mkl-fft-fn (* 2 8192)))

    )

  )
