try:

    from gensim.csparse.psparse import pmultiply as cs_gaxpy

    def gaxpy(X, Y): return cs_gaxpy(X, Y)

    openmp = True

except:

    def gaxpy(X, Y): return X * Y

    openmp = False

