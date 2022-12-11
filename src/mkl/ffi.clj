(ns mkl.ffi
  (:require [tech.v3.datatype.ffi :as dt-ffi]
            [tech.v3.datatype.ffi.size-t :as ffi-size-t]
            [tech.v3.datatype.ffi.ptr-value :as ffi-ptr-value]
            [tech.v3.datatype.native-buffer :as native-buffer]
            [tech.v3.datatype.casting :as casting]))


(dt-ffi/define-library!
  libmkl
  '{:mkl_malloc {:rettype :pointer
                 :argtypes [[byte-size :size-t]
                            [alignment :int64]]
                 :doc "Use mkl's allocator to allocate memory.  Alignment defaults to 64 bit"}
    :mkl_free {:rettype :void
               :argtypes [[ptr :pointer]]}
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
                             [incy :int32]]}}
  nil
  nil)


(defn set-library-instance!
  [lib-instance]
  (dt-ffi/library-singleton-set-instance! libmkl lib-instance))


(defn initialize!
  ([] (initialize! nil))
  ([options]
   (let [libpath (get options :mkl-path "mkl_rt")]
     (dt-ffi/library-singleton-set! libmkl libpath))))
