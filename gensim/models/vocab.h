#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>


struct VocabItem {
    long long sample_int;
    npy_uint32 index;
    npy_uint8* code;
    int code_len;
    npy_uint32* point;
    bool error;
};


int DictContains(PyObject *dict, PyObject *key) {
    return PyDict_Contains(dict, key);
}

VocabItem GetVocabItemFrom(PyObject *dict, PyObject *word, bool hs) {
    VocabItem result;
    PyObject *item = NULL, *attr = NULL;

    result.error = false;
    item = PyDict_GetItem(dict, word);

    // Fill result.sample_int
    if (PyObject_HasAttrString(item, "sample_int") != 1) {
        result.error = true;
        goto error;
    }
    attr = PyObject_GetAttrString(item, "sample_int");
    result.sample_int = PyInt_AsLong(attr);
    Py_XDECREF(attr);
    printf("sample_int OK\n");

    // Fill result.index
    if (PyObject_HasAttrString(item, "index") != 1) {
        result.error = true;
        goto error;
    }
    attr = PyObject_GetAttrString(item, "index");
    result.index = PyInt_AsLong(attr);
    Py_XDECREF(attr);
    printf("index OK\n");

    if (hs) {
        // Fill result.code and result.code_len
        if (PyObject_HasAttrString(item, "code") != 1) {
            result.error = true;
            goto error;
        }
        attr = PyObject_GetAttrString(item, "code");
        if (!PyArray_Check(attr)) {
            result.error = true;
            printf("code is not ndarray!\n");
            goto error;
        }

        result.code = (npy_uint8*) PyArray_DATA((PyArrayObject *) attr);
        result.code_len = PyArray_SIZE((PyArrayObject *) attr);
        Py_XDECREF(attr);
        printf("code and code_len OK\n");

        // Fill result.point
        if (PyObject_HasAttrString(item, "point") != 1) {
            result.error = true;
            goto error;
        }
        attr = PyObject_GetAttrString(item, "point");
        if (!PyArray_Check(attr)) {
            result.error = true;
            printf("point is not ndarray!\n");
            goto error;
        }

        result.point = (npy_uint32*) PyArray_DATA((PyArrayObject *) attr);
        Py_XDECREF(attr);
        printf("point OK\n");
    }

 error:
    Py_XDECREF(attr);

    return result;
}