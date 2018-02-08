#include <Python.h>

#if PY_VERSION_HEX >= 0x03020000

/*
** compatibility with python >= 3.2, which doesn't have CObject anymore
*/
static void * PyCObject_AsVoidPtr(PyObject *obj)
{
    void *ret = PyCapsule_GetPointer(obj, NULL);
    if (ret == NULL) {
        PyErr_Clear();
    }
    return ret;
}

#endif