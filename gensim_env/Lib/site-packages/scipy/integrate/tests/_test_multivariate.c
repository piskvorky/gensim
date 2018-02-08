#include <Python.h>

/*
  Backwards compatibility:
  Python2.2 used LONG_LONG instead of PY_LONG_LONG
*/
#if defined(HAVE_LONG_LONG) && !defined(PY_LONG_LONG)
#define PY_LONG_LONG LONG_LONG
#endif

#ifdef MS_WIN32
#include <windows.h>
#endif

#if defined(MS_WIN32) || defined(__CYGWIN__)
#define EXPORT(x) __declspec(dllexport) x
#else
#define EXPORT(x) x
#endif

#include "math.h"
const double PI = 3.141592653589793238462643383279502884;
EXPORT(double)
_multivariate_typical(int n, double *args)
{
    return cos(args[1] * args[0] - args[2] * sin(args[0])) / PI;
}

EXPORT(double)
_multivariate_indefinite(int n, double *args)
{
    return -exp(-args[0]) * log(args[0]);
}

EXPORT(double)
_multivariate_sin(int n, double *args)
{
    return sin(args[0]);
}

EXPORT(double)
_sin_0(double x, void *user_data)
{
    return sin(x);
}

EXPORT(double)
_sin_1(int ndim, double *x, void *user_data)
{
    return sin(x[0]);
}

EXPORT(double)
_sin_2(double x)
{
    return sin(x);
}

EXPORT(double)
_sin_3(int ndim, double *x)
{
    return sin(x[0]);
}

/*
  This won't allow you to actually use the methods here. It just
  lets you load the module so you can get at the __file__ attribute.
*/

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_test_multivariate",
    NULL,
    -1,
    NULL, /* Empty methods section */
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit__test_multivariate(void)
{
    return PyModule_Create(&moduledef);
}

#else

PyMODINIT_FUNC
init_test_multivariate(void)
{
    Py_InitModule("_test_multivariate", NULL);
}
#endif
