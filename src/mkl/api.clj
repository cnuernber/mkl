(ns mkl.api
  (:require [mkl.ffi :as ffi]
            [tech.v3.datatype :as dt]
            [tech.v3.datatype.ffi :as dt-ffi]
            [tech.v3.datatype.copy :as copy]
            [tech.v3.datatype.casting :as casting]
            [tech.v3.datatype.native-buffer :as nbuf]
            [tech.v3.resource :as resource]
            [tech.v3.tensor :as dtt]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defn initialize!
  ([] (ffi/initialize!))
  ([options] (ffi/initialize! options)))


(defn axpy!
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
  [mult lhs rhs]
  (axpy! mult lhs (nbuf/clone-native rhs)))
