# Dynamic MKL Bindings

[![Clojars Project](https://clojars.org/com.cnuernber/mkl/latest-version.svg)](https://clojars.org/com.cnuernber/mkl)

Minimal, partial MKL bindings using dtype-next's ffi system.


The only meaningfully finished sections are convolutions and random number generation.


Convolutions are significantly faster when using MKL than with any JVM-provided method especially if they
are large enough to require fft-based methods.


Random number generation, however, for uniform it may be at most 2x and many times it isn't that much faster.  The benefit
of using mkl will increase when you need more sophisticated distributions for example gaussian generation is
about 4x when we compare the intel fast mercenne twister stream compared to the apache commons mercenne twister stream.



See [api.clj](src/mkl/api.clj) for fft example.  The random number generation doesn't have an example but it does
have good docs :-).


* [API Docs](cnuernber.github.io/mkl)
