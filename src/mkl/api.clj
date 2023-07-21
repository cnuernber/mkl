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
           [tech.v3.datatype Complex]
           [tech.v3.datatype.ffi Pointer]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defn initialize!
  "Initialize - find the `mkl_rt` shared library and load it.  Uses normal OS pathways to
  find the library in addition to the `java_library_path` system variable."
  ([] (ffi/initialize!))
  ([options] (ffi/initialize! options)))


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
    (sub! a b (dt/alloc-uninitialized adt aec))))


(defn real->complex
  "Convert a real buffer to an interleaved real-complex buffer - not an efficient operation."
  [real]
  (let [rc (dt/ecount real)
        rval (dt/alloc-zeros :float64 (* 2 rc))]
    (resource/stack-resource-context
     (let [rvec (dt/alloc-uninitialized :float64 rc)]
       (dt/copy! real rvec)
       (ffi/vdUnpackI rc rvec rval 2)))
    rval))


(defn fft-forward-fn
  "Create a function to process forward fft.  Returned function takes
  one arg, input and returns a pre-allocated output after each invocation.  Input
  length must match len.  If domain is complex, input  length must be double
  len.

  Complex data is stored packed as if in a struct - r1,i1,r2,i2

  * `dtype` - either `:float32` or `:float64`.
  * `domain` - either `:real` or `:complex`.

  Example:

```clojure
user> (require '[cnuernber.mkl :as mkl])
Execution error (FileNotFoundException) at user/eval5625 (REPL:43).
Could not locate cnuernber/mkl__init.class, cnuernber/mkl.clj or cnuernber/mkl.cljc on classpath.
user> (require '[mkl.api :as mkl])
nil
user> (mkl/initialize!)
#object[com.sun.proxy.$Proxy3 0x56dd4be7 \"Proxy interface to Native Library <libmkl_rt.so@139886755541424>\"]
user> (def data (take 100 (cycle [1 2 3 4 5])))
#'user/data
user> (def fft-fn (mkl/fft-forward-fn :float32 :real 100))
#'user/fft-fnJul 21, 2023 3:27:56 PM clojure.tools.logging$eval5978$fn__5981 invoke
INFO: Reference thread starting

user> (def res (fft-fn data))
#'user/res
user> res
#native-buffer@0x00007F39ED2F22D0<float32>[200]
[300.0, 0.000, 0.000, 0.000, 3.104E-07, -3.306E-07, 0.000, 0.000, 1.885E-07, 3.595E-08, 0.000, 0.000, 1.808E-07, -9.941E-08, 0.000, 0.000, -1.462E-07, 1.847E-08, 0.000, 0.000...]
```"
  [dtype domain ^long len]
  (let [desc (ffi/DftiCreateDescriptor dtype domain len)
        outlen (* 2 len)
        inlen (long (case domain
                      :real len
                      :complex (* 2 len)))
        input (dt/alloc-uninitialized dtype inlen)
        output (dt/alloc-uninitialized dtype outlen)
        filler! (if (identical? domain :real)
                  (let [invec (dt/sub-buffer output 2 (- len 2))
                        inlen (dt/ecount invec)
                        in-real-end (dt/sub-buffer invec (- inlen 2))
                        in-complex-end (dt/sub-buffer invec (- inlen 1))
                        out-real-start (dt/sub-buffer output (+ len 2))
                        out-complex-start (dt/sub-buffer output (+ len 3))
                        neg1 (dt/make-container :native-heap dtype [-1])
                        zero (dt/alloc-zeros dtype 1)
                        n-copy (dec (quot len 2))]
                    (fn []
                      ;;fix complex conjugate for real data
                      (case dtype
                        :float32
                        (do
                          (ffi/vsAddI n-copy in-real-end -2 zero 0 out-real-start 2)
                          (ffi/vsMulI n-copy in-complex-end -2 neg1 0 out-complex-start 2))
                        :float64
                        (do
                          (ffi/vdAddI n-copy in-real-end -2 zero 0 out-real-start 2)
                          (ffi/vdMulI n-copy in-complex-end -2 neg1 0 out-complex-start 2)))))
                  (fn []))]
    (fn [user-in]
      (ffi/DftiComputeForward desc (dt/ensure-native user-in input) output)
      (filler!)
      output)))


(defn fft-forward
  "Compute the forward fft.  Returns a dtype-next array buffer.  For options see [[fft-forward-fn]].
  It is a bit faster to use fft-forward-fn and then use the returned function as this initializes
  an fft context only once and reuses it."
  ([data] (fft-forward data nil))
  ([data options]
   (resource/stack-resource-context
    (->
     ((fft-forward-fn (get options :datatype :float32)
                      (get options :domain :real)
                      (dt/ecount data)) data)
     (dt/->array-buffer)))))


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
        output (dt/alloc-uninitialized dtype full-len)
        input (dt/alloc-zeros dtype full-len)
        sigbuffer (dt/sub-buffer input 0 nsignal)
        kernel (dt/alloc-uninitialized dtype nwin)
        retval (case mode
                 :same (dt/sub-buffer output (quot nwin 2) nsignal)
                 :full output)]
    (-> (fn [sigdata kdata]
          (let [kernel (dt/ensure-native kdata kernel)]
            (dt/copy! sigdata sigbuffer)
            (ffi/check-vsl
             "Error executing correlation: "
             (case dtype
               :float32 (ffi/vslsCorrExec1D hdl kernel 1 input 1 output 1)
               :float64 (ffi/vsldCorrExec1D hdl kernel 1 input 1 output 1))))
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
  ([data kernel] (correlate1d data kernel nil))
  ([data kernel options]
   (resource/stack-resource-context
    (dt/->array-buffer
     ((correlation1d-fn (get options :datatype :float32)
                        (dt/ecount data) (dt/ecount kernel) options) data kernel)))))


(defn all-rngs
  []
  (keys ffi/rngs))

(defn all-distributions
  []
  (->> ffi/distributions
       (map (fn [[dist-name dst-data]]
              [dist-name (:arguments dst-data)]))
       (into {})))


(defn rng-stream
  "Return a function that when called returns a new batch of random numbers.  Returns the same native buffer
  on every call.  The values of (all-distributions) state the name of the optional arguments and their
  order.

  Options:

  Note that the distribution arguments *must* be provided in the optional arguments map.  See the documentation
  for various [mkl distribution functions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/distribution-generators.html) for their definitions.


  * `:datatype` - defaults to :float64.
  * `:seed` - unsigned integer - defaults to unchecked integer cast of system/nanoTime if not provided.
  * `:rng` - one of (all-rngs) - defaults to :sfmt19937
  * `:dist` - one of the keys of (all-distributions).  Defaults to :uniform."
  ([^long n {:keys [seed rng dist datatype]
             :or {rng :sfmt19937
                  dist :uniform
                  datatype :float64}
             :as options}]
   (let [seed (unchecked-int (or seed (System/nanoTime)))
         rng-val (ffi/rngs rng)
         dist-map (ffi/distributions dist)
         _ (when-not dist-map
             (throw (RuntimeException. (str "Unrecognized distribution: " dist))))
         dist-args (:arguments dist-map)
         dist-name (:name dist-map)
         dist-fn (get-in dist-map [:fns datatype])
         _ (when-not dist-fn
             (throw (RuntimeException. (str "Distribution does not have function for datatype:" datatype))))
         stream-ptr (dt-ffi/make-ptr :pointer 0)
         _ (ffi/check-dfti "Failed to create stream: " (ffi/vslNewStream stream-ptr rng-val))
         stream (Pointer. (stream-ptr 0))
         result-vec (dt/alloc-uninitialized datatype n)
         argvec (->> dist-args
                     (mapv (fn [sym]
                             (if-let [argval (get options (keyword (name sym)))]
                               argval
                               (throw (RuntimeException. (format "Failed to find argument %s for dist %s"
                                                                 sym dist)))))))]
     (resource/track
      (case (count argvec)
        1 (fn [] (dist-fn 0 stream n result-vec (argvec 0)) result-vec)
        2 (fn [] (dist-fn 0 stream n result-vec (argvec 0) (argvec 1)) result-vec)
        3 (fn [] (dist-fn 0 stream n result-vec (argvec 0) (argvec 1) (argvec 2)) result-vec)
        4 (fn [] (dist-fn 0 stream n result-vec (argvec 0) (argvec 1) (argvec 2) (argvec 3)) result-vec))
      {:track-type :auto
       :dispose-fn #(ffi/vslDeleteStream stream-ptr)}))))


(comment
  (do
    (def small-src (double-array (take 256 (cycle [1 2 3 4 5]))))
    (def large-src (double-array (take 8192 (cycle [1 2 3 4 5]))))
    (def large-large-src (double-array (take (* 8192 2) (cycle [1 2 3 4 5]))))
    (def xlarge-src (double-array (take (* 8192 64) (cycle [1 2 3 4 5]))))
    (def xlarge-native (dt/make-container :native-heap :float32 xlarge-src))

    (defn jfft-fn
      [^long n]
      (let [fft (DoubleFFT_1D. n)]
        (fn [src-data]
          (let [rval (Complex/realToComplex (dt/->double-array src-data))]
            (.complexForward fft rval)
            rval))))

    (defn jfft
      [data]
      ((jfft-fn (dt/ecount data)) data))

    (defn mkl-fft-fn
      [^long n]
      (let [forward-fn (fft-forward-fn :float64 :complex n)
            res (dt/alloc-uninitialized :float64 (* 2 n))]
        (fn [v]
          (resource/stack-resource-context
           (forward-fn v res)))))

    (def jfft-small (jfft-fn (dt/ecount small-src)))
    (def jfft-large (jfft-fn (dt/ecount large-src)))
    (def jfft-large-large (jfft-fn (dt/ecount large-large-src)))


    (def mkl-fft-small (mkl-fft-fn 256))
    (def mkl-fft-large (mkl-fft-fn 8192))
    (def mkl-fft-large-large (mkl-fft-fn (* 2 8192)))

    )

  )
