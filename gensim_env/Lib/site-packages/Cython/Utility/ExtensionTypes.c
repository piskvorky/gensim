
/////////////// CallNextTpDealloc.proto ///////////////

static void __Pyx_call_next_tp_dealloc(PyObject* obj, destructor current_tp_dealloc);

/////////////// CallNextTpDealloc ///////////////

static void __Pyx_call_next_tp_dealloc(PyObject* obj, destructor current_tp_dealloc) {
    PyTypeObject* type = Py_TYPE(obj);
    /* try to find the first parent type that has a different tp_dealloc() function */
    while (type && type->tp_dealloc != current_tp_dealloc)
        type = type->tp_base;
    while (type && type->tp_dealloc == current_tp_dealloc)
        type = type->tp_base;
    if (type)
        type->tp_dealloc(obj);
}

/////////////// CallNextTpTraverse.proto ///////////////

static int __Pyx_call_next_tp_traverse(PyObject* obj, visitproc v, void *a, traverseproc current_tp_traverse);

/////////////// CallNextTpTraverse ///////////////

static int __Pyx_call_next_tp_traverse(PyObject* obj, visitproc v, void *a, traverseproc current_tp_traverse) {
    PyTypeObject* type = Py_TYPE(obj);
    /* try to find the first parent type that has a different tp_traverse() function */
    while (type && type->tp_traverse != current_tp_traverse)
        type = type->tp_base;
    while (type && type->tp_traverse == current_tp_traverse)
        type = type->tp_base;
    if (type && type->tp_traverse)
        return type->tp_traverse(obj, v, a);
    // FIXME: really ignore?
    return 0;
}

/////////////// CallNextTpClear.proto ///////////////

static void __Pyx_call_next_tp_clear(PyObject* obj, inquiry current_tp_dealloc);

/////////////// CallNextTpClear ///////////////

static void __Pyx_call_next_tp_clear(PyObject* obj, inquiry current_tp_clear) {
    PyTypeObject* type = Py_TYPE(obj);
    /* try to find the first parent type that has a different tp_clear() function */
    while (type && type->tp_clear != current_tp_clear)
        type = type->tp_base;
    while (type && type->tp_clear == current_tp_clear)
        type = type->tp_base;
    if (type && type->tp_clear)
        type->tp_clear(obj);
}

/////////////// SetupReduce.proto ///////////////

static int __Pyx_setup_reduce(PyObject* type_obj);

/////////////// SetupReduce ///////////////
//@requires: ObjectHandling.c::PyObjectGetAttrStr
//@substitute: naming

static int __Pyx_setup_reduce_is_named(PyObject* meth, PyObject* name) {
  int ret;
  PyObject *name_attr;

  name_attr = __Pyx_PyObject_GetAttrStr(meth, PYIDENT("__name__"));
  if (likely(name_attr)) {
      ret = PyObject_RichCompareBool(name_attr, name, Py_EQ);
  } else {
      ret = -1;
  }

  if (unlikely(ret < 0)) {
      PyErr_Clear();
      ret = 0;
  }

  Py_XDECREF(name_attr);
  return ret;
}

static int __Pyx_setup_reduce(PyObject* type_obj) {
    int ret = 0;
    PyObject *object_reduce = NULL;
    PyObject *object_reduce_ex = NULL;
    PyObject *reduce = NULL;
    PyObject *reduce_ex = NULL;
    PyObject *reduce_cython = NULL;
    PyObject *setstate = NULL;
    PyObject *setstate_cython = NULL;

#if CYTHON_USE_PYTYPE_LOOKUP
    if (_PyType_Lookup((PyTypeObject*)type_obj, PYIDENT("__getstate__"))) goto GOOD;
#else
    if (PyObject_HasAttr(type_obj, PYIDENT("__getstate__"))) goto GOOD;
#endif

#if CYTHON_USE_PYTYPE_LOOKUP
    object_reduce_ex = _PyType_Lookup(&PyBaseObject_Type, PYIDENT("__reduce_ex__")); if (!object_reduce_ex) goto BAD;
#else
    object_reduce_ex = __Pyx_PyObject_GetAttrStr((PyObject*)&PyBaseObject_Type, PYIDENT("__reduce_ex__")); if (!object_reduce_ex) goto BAD;
#endif

    reduce_ex = __Pyx_PyObject_GetAttrStr(type_obj, PYIDENT("__reduce_ex__")); if (unlikely(!reduce_ex)) goto BAD;
    if (reduce_ex == object_reduce_ex) {

#if CYTHON_USE_PYTYPE_LOOKUP
        object_reduce = _PyType_Lookup(&PyBaseObject_Type, PYIDENT("__reduce__")); if (!object_reduce) goto BAD;
#else
        object_reduce = __Pyx_PyObject_GetAttrStr((PyObject*)&PyBaseObject_Type, PYIDENT("__reduce__")); if (!object_reduce) goto BAD;
#endif
        reduce = __Pyx_PyObject_GetAttrStr(type_obj, PYIDENT("__reduce__")); if (unlikely(!reduce)) goto BAD;

        if (reduce == object_reduce || __Pyx_setup_reduce_is_named(reduce, PYIDENT("__reduce_cython__"))) {
            reduce_cython = __Pyx_PyObject_GetAttrStr(type_obj, PYIDENT("__reduce_cython__")); if (unlikely(!reduce_cython)) goto BAD;
            ret = PyDict_SetItem(((PyTypeObject*)type_obj)->tp_dict, PYIDENT("__reduce__"), reduce_cython); if (unlikely(ret < 0)) goto BAD;
            ret = PyDict_DelItem(((PyTypeObject*)type_obj)->tp_dict, PYIDENT("__reduce_cython__")); if (unlikely(ret < 0)) goto BAD;

            setstate = __Pyx_PyObject_GetAttrStr(type_obj, PYIDENT("__setstate__"));
            if (!setstate) PyErr_Clear();
            if (!setstate || __Pyx_setup_reduce_is_named(setstate, PYIDENT("__setstate_cython__"))) {
                setstate_cython = __Pyx_PyObject_GetAttrStr(type_obj, PYIDENT("__setstate_cython__")); if (unlikely(!setstate_cython)) goto BAD;
                ret = PyDict_SetItem(((PyTypeObject*)type_obj)->tp_dict, PYIDENT("__setstate__"), setstate_cython); if (unlikely(ret < 0)) goto BAD;
                ret = PyDict_DelItem(((PyTypeObject*)type_obj)->tp_dict, PYIDENT("__setstate_cython__")); if (unlikely(ret < 0)) goto BAD;
            }
            PyType_Modified((PyTypeObject*)type_obj);
        }
    }
    goto GOOD;

BAD:
    if (!PyErr_Occurred())
        PyErr_Format(PyExc_RuntimeError, "Unable to initialize pickling for %s", ((PyTypeObject*)type_obj)->tp_name);
    ret = -1;
GOOD:
#if !CYTHON_USE_PYTYPE_LOOKUP
    Py_XDECREF(object_reduce);
    Py_XDECREF(object_reduce_ex);
#endif
    Py_XDECREF(reduce);
    Py_XDECREF(reduce_ex);
    Py_XDECREF(reduce_cython);
    Py_XDECREF(setstate);
    Py_XDECREF(setstate_cython);
    return ret;
}
