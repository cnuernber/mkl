(ns mkl.ffi
  (:require [tech.v3.datatype.ffi :as dt-ffi]
            [tech.v3.datatype.ffi.size-t :as ffi-size-t]
            [tech.v3.datatype.ffi.ptr-value :as ffi-ptr-value]
            [tech.v3.datatype.ffi.jna :as ffi-jna]
            [tech.v3.datatype.native-buffer :as native-buffer]
            [tech.v3.datatype.casting :as casting]
            [tech.v3.resource :as resource])
  (:import [mkl DFTI]
           [tech.v3.datatype.ffi Pointer]
           [com.sun.jna Native]))

(defn vecop
  [name]
  (->>
   (for [dt ['s 'd 'c 'z]
         indexed? [false true]]
     [(keyword (str "v" dt name (if indexed? "I" "")))
      {:rettype :void
       :argtypes
       (if indexed?
         '[[n :int32]
           [a :pointer] ;;double*
           [inca :int32]
           [b :pointer] ;;double*
           [incb :int32]
           [c :pointer] ;;double*
           [incc :int32]
           ]
         '[[n :int32]
           [a :pointer] ;;double*
           [b :pointer] ;;double*
           [c :pointer] ;;double*
           ]
         )}])
   (into {})))


(def real-distributions
  "These all have same shape"
  {"Beta" '[p q a beta]
   "Cauchy" '[a beta]
   "Exponential" '[a beta]
   "Gamma" '[alpha a beta]
   "Gaussian" '[a sigma]
   "Gumbel" '[a beta]
   "Laplace" '[a beta]
   "Lognormal" '[b beta]
   "Rayleigh" '[a beta]
   "Uniform" '[a b]
   "Weibull" '[alpha a beta]})


(dt-ffi/define-library!
  libmkl
  (merge
   '{
     ;;VERY CRASHYYY!!!!
     ;; :mkl_malloc {:rettype :pointer
     ;;              :argtypes [[byte-size :size-t]
     ;;                         [alignment :int64]]
     ;;              :doc "Use mkl's allocator to allocate memory.  Alignment defaults to 64 bit"}
     ;; :mkl_free {:rettype :void
     ;;            :argtypes [[ptr :pointer]]}


     :cblas_saxpy {:rettype :void
                   :argtypes [[n :int32]
                              [a :float32]
                              [x :pointer];;float*
                              [incx :int32]
                              [y :pointer];;float*
                              [incy :int32]]}
     :cblas_daxpy {:rettype :void
                   :argtypes [[n :int32]
                              [a :float64]
                              [x :pointer];;double*
                              [incx :int32]
                              [y :pointer];;double*
                              [incy :int32]]}
     :vsUnpackI {:rettype :void
                 :argtypes [[n :int32]
                            [a :pointer]
                            [y :pointer]
                            [incy :int32]]}
     :vdUnpackI {:rettype :void
                 :argtypes [[n :int32]
                            [a :pointer]
                            [y :pointer]
                            [incy :int32]]}
     :vslsCorrNewTask1D {:rettype :int32
                         :argtypes [[hdl :pointer]
                                    [mode :int32]
                                    [xshape :int32]
                                    [yshape :int32]
                                    [zshape :int32]
                                    ]}
     :vsldCorrNewTask1D {:rettype :int32
                         :argtypes [[hdl :pointer]
                                    [mode :int32]
                                    [xshape :int32]
                                    [yshape :int32]
                                    [zshape :int32]
                                    ]}
     :vslcCorrNewTask1D {:rettype :int32
                         :argtypes [[hdl :pointer]
                                    [mode :int32]
                                    [xshape :int32]
                                    [yshape :int32]
                                    [zshape :int32]
                                    ]}
     :vslzCorrNewTask1D {:rettype :int32
                         :argtypes [[hdlPtr :pointer]
                                    [mode :int32]
                                    [xshape :int32]
                                    [yshape :int32]
                                    [zshape :int32]
                                    ]}
     :vslsCorrExec1D {:rettype :int32
                      :argtypes [[hdl :pointer]
                                 [x :pointer]
                                 [xinc :int32]
                                 [y :pointer]
                                 [yinc :int32]
                                 [z :pointer]
                                 [zinc :int32]]}
     :vsldCorrExec1D {:rettype :int32
                      :argtypes [[hdl :pointer]
                                 [x :pointer]
                                 [xinc :int32]
                                 [y :pointer]
                                 [yinc :int32]
                                 [z :pointer]
                                 [zinc :int32]]}
     :vslcCorrExec1D {:rettype :int32
                      :argtypes [[hdl :pointer]
                                 [x :pointer]
                                 [xinc :int32]
                                 [y :pointer]
                                 [yinc :int32]
                                 [z :pointer]
                                 [zinc :int32]]}
     :vslzCorrExec1D {:rettype :int32
                      :argtypes [[hdl :pointer]
                                 [x :pointer]
                                 [xinc :int32]
                                 [y :pointer]
                                 [yinc :int32]
                                 [z :pointer]
                                 [zinc :int32]]}
     :vslCorrDeleteTask {:rettype :int32
                         :argtypes [[hdlPtr :pointer]]}
     }
   (vecop "Add")
   (vecop "Sub")
   (vecop "Mul")
   (into '{:vslNewStream {:rettype :int32
                          :argtypes [[stream :pointer] ;;ptrptr
                                     [brng :int32
                                      seed :int32]]}
           :vslDeleteStream {:rettype :int32
                             :argtypes [[stream :pointer]] ;;ptrptr
                             }}
         (->> real-distributions
              (mapcat (fn [kv]
                        (let [[dist args] kv]
                          [[(keyword (str "vsRng" dist))
                            {:rettype :int32
                             :argtypes (vec (concat
                                             '[[method :int32]
                                               [stream :pointer]
                                               [nelems :int32]
                                               [r :pointer]]
                                             (mapv #(vector % :float32) args)))}]
                           [(keyword (str "vdRng" dist))
                            {:rettype :int32
                             :argtypes (vec (concat
                                             '[[method :int32]
                                               [stream :pointer]
                                               [nelems :int32]
                                               [r :pointer]]
                                             (mapv #(vector % :float64) args)))}]])))))
   {:vsRngChiSquare {:rettype :int32
                     :argytpes '[[method :int32]
                                 [stream :pointer]
                                 [nelems :int32]
                                 [r :pointer]
                                 [degfree :int32]]}
    :vdRngChiSquare {:rettype :int32
                     :argytpes '[[method :int32]
                                 [stream :pointer]
                                 [nelems :int32]
                                 [r :pointer]
                                 [degfree :int32]]}})
  nil
  nil)


(defn set-library-instance!
  [lib-instance]
  (dt-ffi/library-singleton-set-instance! libmkl lib-instance))


(defonce Dfti* (atom nil))

(defn Dfti
  ^DFTI []
  (if-let [r @Dfti*]
    r
    (throw (RuntimeException. "Dfti interface is nil - have you called initialize!?"))))


(defn initialize!
  ([] (initialize! nil))
  ([options]
   (let [libpath (get options :mkl-path "mkl_rt")]
     (dt-ffi/library-singleton-set! libmkl libpath)
     (reset! Dfti* (Native/load libpath DFTI)))))


(def DFTI_SINGLE 35)
(def DFTI_DOUBLE 36)
(def DFTI_COMPLEX 32)
(def DFTI_REAL 33)
(def DFTI_PLACEMENT 11)
(def DFTI_NOT_INPLACE 44)
(def DFTI_COMPLEX_STORAGE 8)
(def DFTI_COMPLEX_COMPLEX 39)
(def DFTI_COMPLEX_REAL 40)


(defn precision->dfti
  ^long [precision]
  (long (case precision
          :float32 DFTI_SINGLE
          :float64 DFTI_DOUBLE)))

(defn domain->dfti
  ^long [domain]
  (case domain
    :real DFTI_REAL
    :complex DFTI_COMPLEX))


(defn DftiFreeDescriptor
  ^long [^Pointer hdl]
  (resource/stack-resource-context
   (let [hdl-ptr (dt-ffi/make-ptr :pointer (.address hdl))]
     (.DftiFreeDescriptor (Dfti) (ffi-jna/ptr-value hdl-ptr)))))


(defmacro check-dfti
  [msg & code]
  `(let [res# (do ~@code)]
     (when-not (== 0 res#)
       (throw (RuntimeException. (str ~msg res#))))))


(defn DftiCreateDescriptor
  "Create a DFTI descriptor.

  * `precision` - Either `:float32` or `:float64`.
  * `domain` - Either `:real` or `:complex`.
  * `len` - Length of the input.

  Returns a pointer that must be freed using DftiFreeDescriptor."
  [precision domain len]
  (let [^Pointer rval
        (resource/stack-resource-context
         (let [hdl (dt-ffi/make-ptr :pointer 0)
               rval (.DftiCreateDescriptor (Dfti) (ffi-jna/ptr-value hdl)
                                           (precision->dfti precision)
                                           (domain->dfti domain) 1
                                           (object-array [(long len)]))]
           (Pointer. (long (hdl 0)))))
        jptr (ffi-jna/ptr-value rval)
        addr (.address rval)]
    (check-dfti "Setting placement: "
                (.DftiSetValue (Dfti) jptr DFTI_PLACEMENT
                               (object-array [(int DFTI_NOT_INPLACE)])))
    (when (identical? domain :complex)
      (check-dfti "Setting complex storage: "
                  (.DftiSetValue (Dfti) jptr DFTI_COMPLEX_STORAGE
                                 (object-array [(int DFTI_COMPLEX_COMPLEX)]))))
    (check-dfti "Commiting descriptor: "
                (.DftiCommitDescriptor (Dfti) jptr))
    (resource/track rval {:track-type :auto
                          :dispose-fn #(DftiFreeDescriptor (Pointer. addr))})))


(defn DftiComputeForward
  [hdl input output]
  (check-dfti "Compute forward: "
              (.DftiComputeForward (Dfti) (ffi-jna/ptr-value hdl)
                                   (ffi-jna/ptr-value input)
                                   (object-array [(ffi-jna/ptr-value output)])))
  output)


(defmacro check-vsl
  [msg & code]
  `(let [rval# (do ~@code)]
     (when-not (== rval# 0)
       (throw (RuntimeException. (str ~msg rval#))))))


(def VSL_RNG_METHOD_ACCURACY_FLAG (bit-shift-left 1 30))
(def VSL_RNG_METHOD_STD 0)
(def VSL_RNG_METHOD_STD_ACCURATE (bit-or VSL_RNG_METHOD_ACCURACY_FLAG VSL_RNG_METHOD_STD))
(def VSL_RNG_METHOD_UNIFORMBITS_STD 0)
(def VSL_RNG_METHOD_UNIFORMBITS32_STD 0)
(def VSL_RNG_METHOD_UNIFORMBITS64_STD 0)
(def VSL_RNG_METHOD_GAUSSIAN_BOXMULLER 0)
(def VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2 1)
(def VSL_RNG_METHOD_GAUSSIAN_ICDF 2)
(def VSL_RNG_METHOD_EXPONENTIAL_ICDF 0)
(def VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE (bit-or VSL_RNG_METHOD_ACCURACY_FLAG VSL_RNG_METHOD_EXPONENTIAL_ICDF))


(def VSL_BRNG_SHIFT      20)
(def VSL_BRNG_INC        (bit-shift-left 1 VSL_BRNG_SHIFT))

(def VSL_BRNG_MCG31          VSL_BRNG_INC)
(def VSL_BRNG_R250           (+ VSL_BRNG_MCG31 VSL_BRNG_INC))
(def VSL_BRNG_MRG32K3A       (+ VSL_BRNG_R250 VSL_BRNG_INC))
(def VSL_BRNG_MCG59          (+ VSL_BRNG_MRG32K3A VSL_BRNG_INC))
(def VSL_BRNG_WH             (+ VSL_BRNG_MCG59 VSL_BRNG_INC))
(def VSL_BRNG_SOBOL          (+ VSL_BRNG_WH VSL_BRNG_INC))
(def VSL_BRNG_NIEDERR        (+ VSL_BRNG_SOBOL VSL_BRNG_INC))
(def VSL_BRNG_MT19937        (+ VSL_BRNG_NIEDERR  VSL_BRNG_INC))
(def VSL_BRNG_MT2203         (+ VSL_BRNG_MT19937  VSL_BRNG_INC))
(def VSL_BRNG_IABSTRACT      (+ VSL_BRNG_MT2203 VSL_BRNG_INC))
(def VSL_BRNG_DABSTRACT      (+ VSL_BRNG_IABSTRACT VSL_BRNG_INC))
(def VSL_BRNG_SABSTRACT      (+ VSL_BRNG_DABSTRACT VSL_BRNG_INC))
(def VSL_BRNG_SFMT19937      (+ VSL_BRNG_SABSTRACT VSL_BRNG_INC))
(def VSL_BRNG_NONDETERM      (+ VSL_BRNG_SFMT19937 VSL_BRNG_INC))
(def VSL_BRNG_ARS5           (+ VSL_BRNG_NONDETERM VSL_BRNG_INC))
(def VSL_BRNG_PHILOX4X32X10  (+ VSL_BRNG_ARS5 VSL_BRNG_INC))


(def rngs
  (clojure.set/map-invert {VSL_BRNG_MCG31         :mcg31
                           VSL_BRNG_R250          :r250
                           VSL_BRNG_MRG32K3A      :mrg32k3a
                           VSL_BRNG_MCG59         :mcg59
                           VSL_BRNG_WH            :wh
                           VSL_BRNG_SOBOL         :sobol
                           VSL_BRNG_NIEDERR       :niederr
                           VSL_BRNG_MT19937       :mt19937
                           VSL_BRNG_MT2203        :mt2203
                           VSL_BRNG_IABSTRACT     :iabstract
                           VSL_BRNG_DABSTRACT     :dabstract
                           VSL_BRNG_SABSTRACT     :sabstract
                           VSL_BRNG_SFMT19937     :sfmt19937
                           VSL_BRNG_NONDETERM     :nondeterm
                           VSL_BRNG_ARS5          :ars5
                           VSL_BRNG_PHILOX4X32X10 :philox4x32x10 }))


(def distributions
  (->> (merge real-distributions
              {"ChiSquared" '[degfree]})
       (map (fn [kv]
              [(keyword (.toLowerCase ^String (key kv)))
               {:name (key kv)
                :fns {:float64 (resolve (symbol (str "vdRng" (key kv))))
                      :float32 (resolve (symbol (str "vsRng" (key kv))))}
                :arguments (val kv)}]))
       (into {})))
