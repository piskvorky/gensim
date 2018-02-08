/*
 * The SIP module interface.
 *
 * Copyright (c) 2015 Riverbank Computing Limited <info@riverbankcomputing.com>
 *
 * This file is part of SIP.
 *
 * This copy of SIP is licensed for use under the terms of the SIP License
 * Agreement.  See the file LICENSE for more details.
 *
 * This copy of SIP may also used under the terms of the GNU General Public
 * License v2 or v3 as published by the Free Software Foundation which can be
 * found in the files LICENSE-GPL2 and LICENSE-GPL3 included in this package.
 *
 * SIP is supplied WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */


#ifndef _SIP_H
#define _SIP_H


/*
 * This gets round a problem with Qt's moc and Python v2.3.  Strictly speaking
 * it's a Qt problem but later versions of Python include a fix for it so we
 * might as well too.
 */
#undef slots


#include <Python.h>

/*
 * There is a mis-feature somewhere with the Borland compiler.  This works
 * around it.
 */
#if defined(__BORLANDC__)
#include <rpc.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif


/* Sanity check on the Python version. */
#if PY_VERSION_HEX < 0x02030000
#error "This version of SIP requires Python v2.3 or later"
#endif


/*
 * Define the SIP version number.
 */
#define SIP_VERSION         0x041201
#define SIP_VERSION_STR     "4.18.1"


/*
 * Define the current API version number.  SIP must handle modules with the
 * same major number and with the same or earlier minor number.  Whenever data
 * structure elements are added they must be appended and the minor number
 * incremented.  Whenever data structure elements are removed or the order
 * changed then the major number must be incremented and the minor number set
 * to 0.
 *
 * History:
 *
 * 11.3 Added sip_api_get_interpreter() to the public API.
 *
 * 11.1 Added sip_api_invoke_slot_ex().
 * 11.2 Added sip_api_get_reference() to the private API.
 *
 * 11.1 Added sip_api_invoke_slot_ex().
 *
 * 11.0 Added the pyqt5QtSignal and pyqt5ClassTypeDef structures.
 *      Removed qt_interface from pyqt4ClassTypeDef.
 *      Added hack to pyqt4QtSignal.
 *
 * 10.1 Added ctd_final to sipClassTypeDef.
 *      Added ctd_init_mixin to sipClassTypeDef.
 *      Added sip_api_get_mixin_address() to the public API.
 *      Added sip_api_convert_from_new_pytype() to the public API.
 *      Added sip_api_convert_to_array() to the public API.
 *      Added sip_api_convert_to_typed_array() to the public API.
 *      Added sip_api_register_proxy_resolver() to the public API.
 *      Added sip_api_init_mixin() to the private API.
 *      Added qt_interface to pyqt4ClassTypeDef.
 *
 * 10.0 Added sip_api_set_destroy_on_exit().
 *      Added sip_api_enable_autoconversion().
 *      Removed sip_api_call_error_handler_old().
 *      Removed sip_api_start_thread().
 *
 * 9.2  Added sip_gilstate_t and SIP_RELEASE_GIL to the public API.
 *      Renamed sip_api_call_error_handler() to
 *      sip_api_call_error_handler_old().
 *      Added the new sip_api_call_error_handler() to the private API.
 *
 * 9.1  Added the capsule type.
 *      Added the 'z' format character to sip_api_build_result().
 *      Added the 'z', '!' and '$' format characters to
 *      sip_api_parse_result_ex().
 *
 * 9.0  Changed the sipVariableGetterFunc signature.
 *      Added sip_api_parse_result_ex() to the private API.
 *      Added sip_api_call_error_handler() to the private API.
 *      Added em_virterrorhandlers to sipExportedModuleDef.
 *      Re-ordered the API functions.
 *
 * 8.1  Revised the sipVariableDef structure.
 *      sip_api_get_address() is now part of the public API.
 *
 * 8.0  Changed the size of the sipSimpleWrapper structure.
 *      Added sip_api_get_address().
 *
 * 7.1  Added the 'H' format character to sip_api_parse_result().
 *      Deprecated the 'D' format character of sip_api_parse_result().
 *
 * 7.0  Added sip_api_parse_kwd_args().
 *      Added sipErrorState, sip_api_add_exception().
 *      The type initialisation function is now passed a dictionary of keyword
 *      arguments.
 *      All argument parsers now update a set of error messages rather than an
 *      argument count.
 *      The signatures of sip_api_no_function() and sip_api_no_method() have
 *      changed.
 *      Added ctd_docstring to sipClassTypeDef.
 *      Added vf_docstring to sipVersionedFunctionDef.
 *
 * 6.0  Added the sipContainerDef structure to define the contents of a class
 *      or mapped type.  Restructured sipClassDef and sipMappedTypeDef
 *      accordingly.
 *      Added the 'r' format character to sip_api_parse_args().
 *      Added the 'r' format character to sip_api_call_method() and
 *      sip_api_build_result().
 *      Added the assignment, array and copy allocation helpers.
 *
 * 5.0  Added sip_api_is_api_enabled().
 *      Renamed the td_version_nr member of sipTypeDef to be int and where -1
 *      indicates it is not versioned.
 *      Added the em_versions member to sipExportedModuleDef.
 *      Added the em_versioned_functions member to sipExportedModuleDef.
 *
 * 4.0  Much refactoring.
 *
 * 3.8  Added sip_api_register_qt_metatype() and sip_api_deprecated().
 *      Added qt_register_meta_type() to the Qt support API.
 *      The C/C++ names of enums and types are now always defined in the
 *      relevant structures and don't default to the Python name.
 *      Added the 'XE' format characters to sip_api_parse_args().
 *
 * 3.7  Added sip_api_convert_from_const_void_ptr(),
 *      sip_api_convert_from_void_ptr_and_size() and
 *      sip_api_convert_from_const_void_ptr_and_size().
 *      Added the 'g' and 'G' format characters (to replace the now deprecated
 *      'a' and 'A' format characters) to sip_api_build_result(),
 *      sip_api_call_method() and sip_api_parse_result().
 *      Added the 'k' and 'K' format characters (to replace the now deprecated
 *      'a' and 'A' format characters) to sip_api_parse_args().
 *      Added sip_api_invoke_slot().
 *      Added sip_api_parse_type().
 *      Added sip_api_is_exact_wrapped_type().
 *      Added sip_api_assign_instance().
 *      Added sip_api_assign_mapped_type().
 *      Added the td_assign and td_qt fields to the sipTypeDef structure.
 *      Added the mt_assign field to the sipMappedType structure.
 *
 * 3.6  Added the 'g' format character to sip_api_parse_args().
 *
 * 3.5  Added the td_pickle field to the sipTypeDef structure.
 *      Added sip_api_transfer_break().
 *
 * 3.4  Added qt_find_connection() to the Qt support API.
 *      Added sip_api_string_as_char(), sip_api_unicode_as_wchar(),
 *      sip_api_unicode_as_wstring(), sip_api_find_class(),
 *      sip_api_find_named_enum() and sip_api_parse_signature().
 *      Added the 'A', 'w' and 'x' format characters to sip_api_parse_args(),
 *      sip_api_parse_result(), sip_api_build_result() and
 *      sip_api_call_method().
 *
 * 3.3  Added sip_api_register_int_types().
 *
 * 3.2  Added sip_api_export_symbol() and sip_api_import_symbol().
 *
 * 3.1  Added sip_api_add_mapped_type_instance().
 *
 * 3.0  Moved the Qt support out of the sip module and into PyQt.  This is
 *      such a dramatic change that there is no point in attempting to maintain
 *      backwards compatibility.
 *
 * 2.0  Added the td_flags field to the sipTypeDef structure.
 *      Added the first_child, sibling_next, sibling_prev and parent fields to
 *      the sipWrapper structure.
 *      Added the td_traverse and td_clear fields to the sipTypeDef structure.
 *      Added the em_api_minor field to the sipExportedModuleDef structure.
 *      Added sip_api_bad_operator_arg().
 *      Added sip_api_wrapper_check().
 *
 * 1.1  Added support for __pos__ and __abs__.
 *
 * 1.0  Removed all deprecated parts of the API.
 *      Removed the td_proxy field from the sipTypeDef structure.
 *      Removed the create proxy function from the 'q' and 'y' format
 *      characters to sip_api_parse_args().
 *      Removed sip_api_emit_to_slot().
 *      Reworked the enum related structures.
 *
 * 0.2  Added the 'H' format character to sip_api_parse_args().
 *
 * 0.1  Added sip_api_add_class_instance().
 *      Added the 't' format character to sip_api_parse_args().
 *      Deprecated the 'J' and 'K' format characters to sip_api_parse_result().
 *
 * 0.0  Original version.
 */
#define SIP_API_MAJOR_NR    11
#define SIP_API_MINOR_NR    3


/* The name of the sip module. */
#define SIP_MODULE_NAME     "sip"


/*
 * Qt includes this typedef and its meta-object system explicitly converts
 * types to uint.  If these correspond to signal arguments then that conversion
 * is exposed.  Therefore SIP generates code that uses it.  This definition is
 * for the cases that SIP is generating non-Qt related bindings with compilers
 * that don't include it themselves (i.e. MSVC).
 */
typedef unsigned int uint;


/* Some Python compatibility stuff. */
#if PY_VERSION_HEX >= 0x02050000

#define SIP_SSIZE_T         Py_ssize_t
#define SIP_SSIZE_T_FORMAT  "%zd"

#define SIP_MLNAME_CAST(s)  (s)
#define SIP_MLDOC_CAST(s)   (s)
#define SIP_TPNAME_CAST(s)  (s)

#else

#define SIP_SSIZE_T         int
#define SIP_SSIZE_T_FORMAT  "%d"

#define SIP_MLNAME_CAST(s)  ((char *)(s))
#define SIP_MLDOC_CAST(s)   ((char *)(s))
#define SIP_TPNAME_CAST(s)  ((char *)(s))

#endif

#if PY_MAJOR_VERSION >= 3

#define SIPLong_FromLong    PyLong_FromLong
#define SIPLong_AsLong      PyLong_AsLong

#define SIPBytes_Check      PyBytes_Check
#define SIPBytes_FromString PyBytes_FromString
#define SIPBytes_FromStringAndSize  PyBytes_FromStringAndSize
#define SIPBytes_AS_STRING  PyBytes_AS_STRING
#define SIPBytes_GET_SIZE   PyBytes_GET_SIZE

#if PY_MINOR_VERSION >= 1
#define SIP_USE_PYCAPSULE
#endif

#if PY_MINOR_VERSION < 2
#define SIP_SUPPORT_PYCOBJECT
#endif

#else

#define SIPLong_FromLong    PyInt_FromLong
#define SIPLong_AsLong      PyInt_AsLong

#define SIPBytes_Check      PyString_Check
#define SIPBytes_FromString PyString_FromString
#define SIPBytes_FromStringAndSize  PyString_FromStringAndSize
#define SIPBytes_AS_STRING  PyString_AS_STRING
#define SIPBytes_GET_SIZE   PyString_GET_SIZE

#if PY_MINOR_VERSION >= 7
#define SIP_USE_PYCAPSULE
#endif

#define SIP_SUPPORT_PYCOBJECT

#endif

#if !defined(Py_REFCNT)
#define Py_REFCNT(ob)       (((PyObject*)(ob))->ob_refcnt)
#endif

#if !defined(Py_TYPE)
#define Py_TYPE(ob)         (((PyObject*)(ob))->ob_type)
#endif

#if !defined(PyVarObject_HEAD_INIT)
#define PyVarObject_HEAD_INIT(type, size)   PyObject_HEAD_INIT(type) size,
#endif


#if defined(SIP_USE_PYCAPSULE)
#define SIPCapsule_FromVoidPtr(p, n)    PyCapsule_New((p), (n), NULL)
#define SIPCapsule_AsVoidPtr(p, n)      PyCapsule_GetPointer((p), (n))
#else
#define SIPCapsule_FromVoidPtr(p, n)    sipConvertFromVoidPtr((p))
#define SIPCapsule_AsVoidPtr(p, n)      sipConvertToVoidPtr((p))
#endif


/*
 * The mask that can be passed to sipTrace().
 */
#define SIP_TRACE_CATCHERS  0x0001
#define SIP_TRACE_CTORS     0x0002
#define SIP_TRACE_DTORS     0x0004
#define SIP_TRACE_INITS     0x0008
#define SIP_TRACE_DEALLOCS  0x0010
#define SIP_TRACE_METHODS   0x0020


/*
 * Hide some thread dependent stuff.
 */
#ifdef WITH_THREAD
typedef PyGILState_STATE sip_gilstate_t;
#define SIP_RELEASE_GIL(gs) PyGILState_Release(gs);
#define SIP_BLOCK_THREADS   {PyGILState_STATE sipGIL = PyGILState_Ensure();
#define SIP_UNBLOCK_THREADS PyGILState_Release(sipGIL);}
#else
typedef int sip_gilstate_t;
#define SIP_RELEASE_GIL(gs)
#define SIP_BLOCK_THREADS
#define SIP_UNBLOCK_THREADS
#endif


/*
 * Some convenient function pointers.
 */

/*
 * The operation an access function is being asked to perform.
 */
typedef enum
{
    UnguardedPointer,   /* Return the unguarded pointer. */
    GuardedPointer,     /* Return the guarded pointer, ie. 0 if it has gone. */
    ReleaseGuard        /* Release the guard, if any. */
} AccessFuncOp;

struct _sipSimpleWrapper;
struct _sipTypeDef;

typedef void *(*sipInitFunc)(struct _sipSimpleWrapper *, PyObject *,
        PyObject *, PyObject **, PyObject **, PyObject **);
typedef int (*sipFinalFunc)(PyObject *, void *, PyObject *, PyObject **);
typedef void *(*sipAccessFunc)(struct _sipSimpleWrapper *, AccessFuncOp);
typedef int (*sipTraverseFunc)(void *, visitproc, void *);
typedef int (*sipClearFunc)(void *);
#if PY_MAJOR_VERSION >= 3
typedef int (*sipGetBufferFunc)(PyObject *, void *, Py_buffer *, int);
typedef void (*sipReleaseBufferFunc)(PyObject *, void *, Py_buffer *);
#else
typedef SIP_SSIZE_T (*sipBufferFunc)(PyObject *, void *, SIP_SSIZE_T, void **);
typedef SIP_SSIZE_T (*sipSegCountFunc)(PyObject *, void *, SIP_SSIZE_T *);
#endif
typedef void (*sipDeallocFunc)(struct _sipSimpleWrapper *);
typedef void *(*sipCastFunc)(void *, const struct _sipTypeDef *);
typedef const struct _sipTypeDef *(*sipSubClassConvertFunc)(void **);
typedef int (*sipConvertToFunc)(PyObject *, void **, int *, PyObject *);
typedef PyObject *(*sipConvertFromFunc)(void *, PyObject *);
typedef void (*sipVirtErrorHandlerFunc)(struct _sipSimpleWrapper *,
        sip_gilstate_t);
typedef int (*sipVirtHandlerFunc)(sip_gilstate_t, sipVirtErrorHandlerFunc,
        struct _sipSimpleWrapper *, PyObject *, ...);
typedef void (*sipAssignFunc)(void *, SIP_SSIZE_T, const void *);
typedef void *(*sipArrayFunc)(SIP_SSIZE_T);
typedef void *(*sipCopyFunc)(const void *, SIP_SSIZE_T);
typedef void (*sipReleaseFunc)(void *, int);
typedef PyObject *(*sipPickleFunc)(void *);
typedef int (*sipAttrGetterFunc)(const struct _sipTypeDef *, PyObject *);
typedef PyObject *(*sipVariableGetterFunc)(void *, PyObject *, PyObject *);
typedef int (*sipVariableSetterFunc)(void *, PyObject *, PyObject *);
typedef void *(*sipProxyResolverFunc)(void *);


/*
 * The meta-type of a wrapper type.
 */
typedef struct _sipWrapperType {
    /*
     * The super-metatype.  This must be first in the structure so that it can
     * be cast to a PyTypeObject *.
     */
    PyHeapTypeObject super;

    /* The generated type structure. */
    struct _sipTypeDef *type;

    /* The list of init extenders. */
    struct _sipInitExtenderDef *iextend;

    /* Set if the type's dictionary contains all lazy attributes. */
    int dict_complete;
} sipWrapperType;


/*
 * The type of a simple C/C++ wrapper object.
 */
typedef struct _sipSimpleWrapper {
    PyObject_HEAD

    /*
     * The data, initially a pointer to the C/C++ object, as interpreted by the
     * access function.
     */
    void *data;

    /* The optional access function. */
    sipAccessFunc access_func;

    /* Object flags. */
    int flags;

    /* The optional dictionary of extra references keyed by argument number. */
    PyObject *extra_refs;

    /* For the user to use. */
    PyObject *user;

    /* The instance dictionary. */
    PyObject *dict;

    /* The main instance if this is a mixin. */
    PyObject *mixin_main;

    /* Next object at this address. */
    struct _sipSimpleWrapper *next;
} sipSimpleWrapper;


/*
 * The type of a C/C++ wrapper object that supports parent/child relationships.
 */
typedef struct _sipWrapper {
    /* The super-type. */
    sipSimpleWrapper super;

    /* First child object. */
    struct _sipWrapper *first_child;

    /* Next sibling. */
    struct _sipWrapper *sibling_next;

    /* Previous sibling. */
    struct _sipWrapper *sibling_prev;

    /* Owning object. */
    struct _sipWrapper *parent;
} sipWrapper;


/*
 * The meta-type of an enum type.  (This is exposed only to support the
 * deprecated sipConvertFromNamedEnum() macro.)
 */
typedef struct _sipEnumTypeObject {
    /*
     * The super-metatype.  This must be first in the structure so that it can
     * be cast to a PyTypeObject *.
     */
    PyHeapTypeObject super;

    /* The generated type structure. */
    struct _sipTypeDef *type;
} sipEnumTypeObject;


/*
 * The information describing an encoded type ID.
 */
typedef struct _sipEncodedTypeDef {
    /* The type number. */
    unsigned sc_type:16;

    /* The module number (255 for this one). */
    unsigned sc_module:8;

    /* A context specific flag. */
    unsigned sc_flag:1;
} sipEncodedTypeDef;


/*
 * The information describing an enum member.
 */
typedef struct _sipEnumMemberDef {
    /* The member name. */
    const char *em_name;

    /* The member value. */
    int em_val;

    /* The member enum, -ve if anonymous. */
    int em_enum;
} sipEnumMemberDef;


/*
 * The information describing static instances.
 */
typedef struct _sipInstancesDef {
    /* The types. */
    struct _sipTypeInstanceDef *id_type;

    /* The void *. */
    struct _sipVoidPtrInstanceDef *id_voidp;

    /* The chars. */
    struct _sipCharInstanceDef *id_char;

    /* The strings. */
    struct _sipStringInstanceDef *id_string;

    /* The ints. */
    struct _sipIntInstanceDef *id_int;

    /* The longs. */
    struct _sipLongInstanceDef *id_long;

    /* The unsigned longs. */
    struct _sipUnsignedLongInstanceDef *id_ulong;

    /* The long longs. */
    struct _sipLongLongInstanceDef *id_llong;

    /* The unsigned long longs. */
    struct _sipUnsignedLongLongInstanceDef *id_ullong;

    /* The doubles. */
    struct _sipDoubleInstanceDef *id_double;
} sipInstancesDef;


/*
 * The information describing a type initialiser extender.
 */
typedef struct _sipInitExtenderDef {
    /* The API version range index. */
    int ie_api_range;

    /* The extender function. */
    sipInitFunc ie_extender;

    /* The class being extended. */
    sipEncodedTypeDef ie_class;

    /* The next extender for this class. */
    struct _sipInitExtenderDef *ie_next;
} sipInitExtenderDef;


/*
 * The information describing a sub-class convertor.
 */
typedef struct _sipSubClassConvertorDef {
    /* The convertor. */
    sipSubClassConvertFunc scc_convertor;

    /* The encoded base type. */
    sipEncodedTypeDef scc_base;

    /* The base type. */
    struct _sipTypeDef *scc_basetype;
} sipSubClassConvertorDef;


/*
 * The different error states of handwritten code.
 */
typedef enum {
    sipErrorNone,       /* There is no error. */
    sipErrorFail,       /* The error is a failure. */
    sipErrorContinue    /* It may not apply if a later operation succeeds. */
} sipErrorState;


/*
 * The different Python slot types.  New slots must be added to the end,
 * otherwise the major version of the internal ABI must be changed.
 */
typedef enum {
    str_slot,           /* __str__ */
    int_slot,           /* __int__ */
#if PY_MAJOR_VERSION < 3
    long_slot,          /* __long__ */
#endif
    float_slot,         /* __float__ */
    len_slot,           /* __len__ */
    contains_slot,      /* __contains__ */
    add_slot,           /* __add__ for number */
    concat_slot,        /* __add__ for sequence types */
    sub_slot,           /* __sub__ */
    mul_slot,           /* __mul__ for number types */
    repeat_slot,        /* __mul__ for sequence types */
    div_slot,           /* __div__ */
    mod_slot,           /* __mod__ */
    floordiv_slot,      /* __floordiv__ */
    truediv_slot,       /* __truediv__ */
    and_slot,           /* __and__ */
    or_slot,            /* __or__ */
    xor_slot,           /* __xor__ */
    lshift_slot,        /* __lshift__ */
    rshift_slot,        /* __rshift__ */
    iadd_slot,          /* __iadd__ for number types */
    iconcat_slot,       /* __iadd__ for sequence types */
    isub_slot,          /* __isub__ */
    imul_slot,          /* __imul__ for number types */
    irepeat_slot,       /* __imul__ for sequence types */
    idiv_slot,          /* __idiv__ */
    imod_slot,          /* __imod__ */
    ifloordiv_slot,     /* __ifloordiv__ */
    itruediv_slot,      /* __itruediv__ */
    iand_slot,          /* __iand__ */
    ior_slot,           /* __ior__ */
    ixor_slot,          /* __ixor__ */
    ilshift_slot,       /* __ilshift__ */
    irshift_slot,       /* __irshift__ */
    invert_slot,        /* __invert__ */
    call_slot,          /* __call__ */
    getitem_slot,       /* __getitem__ */
    setitem_slot,       /* __setitem__ */
    delitem_slot,       /* __delitem__ */
    lt_slot,            /* __lt__ */
    le_slot,            /* __le__ */
    eq_slot,            /* __eq__ */
    ne_slot,            /* __ne__ */
    gt_slot,            /* __gt__ */
    ge_slot,            /* __ge__ */
#if PY_MAJOR_VERSION < 3
    cmp_slot,           /* __cmp__ */
#endif
    bool_slot,          /* __bool__, __nonzero__ */
    neg_slot,           /* __neg__ */
    repr_slot,          /* __repr__ */
    hash_slot,          /* __hash__ */
    pos_slot,           /* __pos__ */
    abs_slot,           /* __abs__ */
#if PY_VERSION_HEX >= 0x02050000
    index_slot,         /* __index__ */
#endif
    iter_slot,          /* __iter__ */
    next_slot,          /* __next__ */
    setattr_slot,       /* __setattr__, __delattr__ */
    matmul_slot,        /* __matmul__ (for Python v3.5 and later) */
    imatmul_slot,       /* __imatmul__ (for Python v3.5 and later) */
    await_slot,         /* __await__ (for Python v3.5 and later) */
    aiter_slot,         /* __aiter__ (for Python v3.5 and later) */
    anext_slot,         /* __anext__ (for Python v3.5 and later) */
} sipPySlotType;


/*
 * The information describing a Python slot function.
 */
typedef struct _sipPySlotDef {
    /* The function. */
    void *psd_func;

    /* The type. */
    sipPySlotType psd_type;
} sipPySlotDef;


/*
 * The information describing a Python slot extender.
 */
typedef struct _sipPySlotExtenderDef {
    /* The function. */
    void *pse_func;

    /* The type. */
    sipPySlotType pse_type;

    /* The encoded class. */
    sipEncodedTypeDef pse_class;
} sipPySlotExtenderDef;


/*
 * The information describing a typedef.
 */
typedef struct _sipTypedefDef {
    /* The typedef name. */
    const char *tdd_name;

    /* The typedef value. */
    const char *tdd_type_name;
} sipTypedefDef;


/*
 * The information describing a variable or property.
 */

typedef enum
{
    PropertyVariable,       /* A property. */
    InstanceVariable,       /* An instance variable. */
    ClassVariable           /* A class (i.e. static) variable. */
} sipVariableType;

typedef struct _sipVariableDef {
    /* The type of variable. */
    sipVariableType vd_type;

    /* The name. */
    const char *vd_name;

    /*
     * The getter.  If this is a variable (rather than a property) then the
     * actual type is sipVariableGetterFunc.
     */
    PyMethodDef *vd_getter;

    /*
     * The setter.  If this is a variable (rather than a property) then the
     * actual type is sipVariableSetterFunc.  It is NULL if the property cannot
     * be set or the variable is const.
     */
    PyMethodDef *vd_setter;

    /* The property deleter. */
    PyMethodDef *vd_deleter;

    /* The docstring. */
    const char *vd_docstring;
} sipVariableDef;


/*
 * The information describing a type, either a C++ class (or C struct), a C++
 * namespace, a mapped type or a named enum.
 */
typedef struct _sipTypeDef {
    /* The version range index, -1 if the type isn't versioned. */
    int td_version;

    /* The next version of this type. */
    struct _sipTypeDef *td_next_version;

    /* The module, 0 if the type hasn't been initialised. */
    struct _sipExportedModuleDef *td_module;

    /* Type flags, see the sipType*() macros. */
    int td_flags;

    /* The C/C++ name of the type. */
    int td_cname;

    /*
     * The Python type object.  This needs to be a union until we remove the
     * deprecated sipClass_* macros.
     */
    union {
        PyTypeObject *td_py_type;
        sipWrapperType *td_wrapper_type;
    } u;
} sipTypeDef;


/*
 * The information describing a container (ie. a class, namespace or a mapped
 * type).
 */
typedef struct _sipContainerDef {
    /*
     * The Python name of the type, -1 if this is a namespace extender (in the
     * context of a class) or doesn't require a namespace (in the context of a
     * mapped type). */
    int cod_name;

    /*
     * The scoping type or the namespace this is extending if it is a namespace
     * extender.
     */
    sipEncodedTypeDef cod_scope;

    /* The number of lazy methods. */
    int cod_nrmethods;

    /* The table of lazy methods. */
    PyMethodDef *cod_methods;

    /* The number of lazy enum members. */
    int cod_nrenummembers;

    /* The table of lazy enum members. */
    sipEnumMemberDef *cod_enummembers;

    /* The number of variables. */
    int cod_nrvariables;

    /* The table of variables. */
    sipVariableDef *cod_variables;

    /* The static instances. */
    sipInstancesDef cod_instances;
} sipContainerDef;


/*
 * The information describing a C++ class (or C struct) or a C++ namespace.
 */
typedef struct _sipClassTypeDef {
    /* The base type information. */
    sipTypeDef ctd_base;

    /* The container information. */
    sipContainerDef ctd_container;

    /* The docstring. */
    const char *ctd_docstring;

    /*
     * The meta-type name, -1 to use the meta-type of the first super-type
     * (normally sipWrapperType).
     */
    int ctd_metatype;

    /* The super-type name, -1 to use sipWrapper. */
    int ctd_supertype;

    /* The super-types. */
    sipEncodedTypeDef *ctd_supers;

    /* The table of Python slots. */
    sipPySlotDef *ctd_pyslots;

    /* The initialisation function. */
    sipInitFunc ctd_init;

    /* The traverse function. */
    sipTraverseFunc ctd_traverse;

    /* The clear function. */
    sipClearFunc ctd_clear;

#if PY_MAJOR_VERSION >= 3
    /* The get buffer function. */
    sipGetBufferFunc ctd_getbuffer;

    /* The release buffer function. */
    sipReleaseBufferFunc ctd_releasebuffer;
#else
    /* The read buffer function. */
    sipBufferFunc ctd_readbuffer;

    /* The write buffer function. */
    sipBufferFunc ctd_writebuffer;

    /* The segment count function. */
    sipSegCountFunc ctd_segcount;

    /* The char buffer function. */
    sipBufferFunc ctd_charbuffer;
#endif

    /* The deallocation function. */
    sipDeallocFunc ctd_dealloc;

    /* The optional assignment function. */
    sipAssignFunc ctd_assign;

    /* The optional array allocation function. */
    sipArrayFunc ctd_array;

    /* The optional copy function. */
    sipCopyFunc ctd_copy;

    /* The release function, 0 if a C struct. */
    sipReleaseFunc ctd_release;

    /* The cast function, 0 if a C struct. */
    sipCastFunc ctd_cast;

    /* The optional convert to function. */
    sipConvertToFunc ctd_cto;

    /* The optional convert from function. */
    sipConvertFromFunc ctd_cfrom;

    /* The next namespace extender. */
    struct _sipClassTypeDef *ctd_nsextender;

    /* The pickle function. */
    sipPickleFunc ctd_pickle;

    /* The finalisation function. */
    sipFinalFunc ctd_final;

    /* The mixin initialisation function. */
    initproc ctd_init_mixin;
} sipClassTypeDef;


/*
 * The information describing a mapped type.
 */
typedef struct _sipMappedTypeDef {
    /* The base type information. */
    sipTypeDef mtd_base;

    /* The container information. */
    sipContainerDef mtd_container;

    /* The optional assignment function. */
    sipAssignFunc mtd_assign;

    /* The optional array allocation function. */
    sipArrayFunc mtd_array;

    /* The optional copy function. */
    sipCopyFunc mtd_copy;

    /* The optional release function. */
    sipReleaseFunc mtd_release;

    /* The convert to function. */
    sipConvertToFunc mtd_cto;

    /* The convert from function. */
    sipConvertFromFunc mtd_cfrom;
} sipMappedTypeDef;


/*
 * The information describing a named enum.
 */
typedef struct _sipEnumTypeDef {
    /* The base type information. */
    sipTypeDef etd_base;

    /* The Python name of the enum. */
    int etd_name;

    /* The scoping type, -1 if it is defined at the module level. */
    int etd_scope;

    /* The Python slots. */
    struct _sipPySlotDef *etd_pyslots;
} sipEnumTypeDef;


/*
 * The information describing an external type.
 */
typedef struct _sipExternalTypeDef {
    /* The index into the type table. */
    int et_nr;

    /* The name of the type. */
    const char *et_name;
} sipExternalTypeDef;


/*
 * The information describing a mapped class.  This (and anything that uses it)
 * is deprecated.
 */
typedef sipTypeDef sipMappedType;


/*
 * Defines an entry in the module specific list of delayed dtor calls.
 */
typedef struct _sipDelayedDtor {
    /* The C/C++ instance. */
    void *dd_ptr;

    /* The class name. */
    const char *dd_name;

    /* Non-zero if dd_ptr is a derived class instance. */
    int dd_isderived;

    /* Next in the list. */
    struct _sipDelayedDtor *dd_next;
} sipDelayedDtor;


/*
 * Defines an entry in the table of global functions all of whose overloads
 * are versioned (so their names can't be automatically added to the module
 * dictionary).
 */
typedef struct _sipVersionedFunctionDef {
    /* The name, -1 marks the end of the table. */
    int vf_name;

    /* The function itself. */
    PyCFunction vf_function;

    /* The METH_* flags. */
    int vf_flags;

    /* The docstring. */
    const char *vf_docstring;

    /* The API version range index. */
    int vf_api_range;
} sipVersionedFunctionDef;


/*
 * The information describing an imported module.
 */
typedef struct _sipImportedModuleDef {
    /* The module name. */
    const char *im_name;

    /* The required version. */
    int im_version;

    /* The imported module. */
    struct _sipExportedModuleDef *im_module;
} sipImportedModuleDef;


/*
 * The main client module structure.
 */
typedef struct _sipExportedModuleDef {
    /* The next in the list. */
    struct _sipExportedModuleDef *em_next;

    /* The SIP API minor version number. */
    unsigned em_api_minor;

    /* The module name. */
    int em_name;

    /* The module name as an object. */
    PyObject *em_nameobj;

    /* The module version. */
    int em_version;

    /* The string pool. */
    const char *em_strings;

    /* The imported modules. */
    sipImportedModuleDef *em_imports;

    /* The optional Qt support API. */
    struct _sipQtAPI *em_qt_api;

    /* The number of types. */
    int em_nrtypes;

    /* The table of types. */
    sipTypeDef **em_types;

    /* The table of external types. */
    sipExternalTypeDef *em_external;

    /* The number of members in global enums. */
    int em_nrenummembers;

    /* The table of members in global enums. */
    sipEnumMemberDef *em_enummembers;

    /* The number of typedefs. */
    int em_nrtypedefs;

    /* The table of typedefs. */
    sipTypedefDef *em_typedefs;

    /* The table of virtual handlers. */
    sipVirtHandlerFunc *em_virthandlers;

    /* The table of virtual error handlers. */
    sipVirtErrorHandlerFunc *em_virterrorhandlers;

    /* The sub-class convertors. */
    sipSubClassConvertorDef *em_convertors;

    /* The static instances. */
    sipInstancesDef em_instances;

    /* The license. */
    struct _sipLicenseDef *em_license;

    /* The table of exception types. */
    PyObject **em_exceptions;

    /* The table of Python slot extenders. */
    sipPySlotExtenderDef *em_slotextend;

    /* The table of initialiser extenders. */
    sipInitExtenderDef *em_initextend;

    /* The delayed dtor handler. */
    void (*em_delayeddtors)(const sipDelayedDtor *);

    /* The list of delayed dtors. */
    sipDelayedDtor *em_ddlist;

    /*
     * The array of API version definitions.  Each definition takes up 3
     * elements.  If the third element of a 3-tuple is negative then the first
     * two elements define an API and its default version.  All such
     * definitions will appear at the end of the array.  If the first element
     * of a 3-tuple is negative then that is the last element of the array.
     */
    int *em_versions;

    /* The optional table of versioned functions. */
    sipVersionedFunctionDef *em_versioned_functions;
} sipExportedModuleDef;


/*
 * The information describing a license to be added to a dictionary.
 */
typedef struct _sipLicenseDef {
    /* The type of license. */
    const char *lc_type;

    /* The licensee. */
    const char *lc_licensee;

    /* The timestamp. */
    const char *lc_timestamp;

    /* The signature. */
    const char *lc_signature;
} sipLicenseDef;


/*
 * The information describing a void pointer instance to be added to a
 * dictionary.
 */
typedef struct _sipVoidPtrInstanceDef {
    /* The void pointer name. */
    const char *vi_name;

    /* The void pointer value. */
    void *vi_val;
} sipVoidPtrInstanceDef;


/*
 * The information describing a char instance to be added to a dictionary.
 */
typedef struct _sipCharInstanceDef {
    /* The char name. */
    const char *ci_name;

    /* The char value. */
    char ci_val;

    /* The encoding used, either 'A', 'L', '8' or 'N'. */
    char ci_encoding;
} sipCharInstanceDef;


/*
 * The information describing a string instance to be added to a dictionary.
 */
typedef struct _sipStringInstanceDef {
    /* The string name. */
    const char *si_name;

    /* The string value. */
    const char *si_val;

    /* The encoding used, either 'A', 'L', '8' or 'N'. */
    char si_encoding;
} sipStringInstanceDef;


/*
 * The information describing an int instance to be added to a dictionary.
 */
typedef struct _sipIntInstanceDef {
    /* The int name. */
    const char *ii_name;

    /* The int value. */
    int ii_val;
} sipIntInstanceDef;


/*
 * The information describing a long instance to be added to a dictionary.
 */
typedef struct _sipLongInstanceDef {
    /* The long name. */
    const char *li_name;

    /* The long value. */
    long li_val;
} sipLongInstanceDef;


/*
 * The information describing an unsigned long instance to be added to a
 * dictionary.
 */
typedef struct _sipUnsignedLongInstanceDef {
    /* The unsigned long name. */
    const char *uli_name;

    /* The unsigned long value. */
    unsigned long uli_val;
} sipUnsignedLongInstanceDef;


/*
 * The information describing a long long instance to be added to a dictionary.
 */
typedef struct _sipLongLongInstanceDef {
    /* The long long name. */
    const char *lli_name;

    /* The long long value. */
#if defined(HAVE_LONG_LONG)
    PY_LONG_LONG lli_val;
#else
    long lli_val;
#endif
} sipLongLongInstanceDef;


/*
 * The information describing an unsigned long long instance to be added to a
 * dictionary.
 */
typedef struct _sipUnsignedLongLongInstanceDef {
    /* The unsigned long long name. */
    const char *ulli_name;

    /* The unsigned long long value. */
#if defined(HAVE_LONG_LONG)
    unsigned PY_LONG_LONG ulli_val;
#else
    unsigned long ulli_val;
#endif
} sipUnsignedLongLongInstanceDef;


/*
 * The information describing a double instance to be added to a dictionary.
 */
typedef struct _sipDoubleInstanceDef {
    /* The double name. */
    const char *di_name;

    /* The double value. */
    double di_val;
} sipDoubleInstanceDef;


/*
 * The information describing a class or enum instance to be added to a
 * dictionary.
 */
typedef struct _sipTypeInstanceDef {
    /* The type instance name. */
    const char *ti_name;

    /* The actual instance. */
    void *ti_ptr;

    /* A pointer to the generated type. */
    struct _sipTypeDef **ti_type;

    /* The wrapping flags. */
    int ti_flags;
} sipTypeInstanceDef;


/*
 * Define a mapping between a wrapped type identified by a string and the
 * corresponding Python type.  This is deprecated.
 */
typedef struct _sipStringTypeClassMap {
    /* The type as a string. */
    const char *typeString;

    /* A pointer to the Python type. */
    struct _sipWrapperType **pyType;
} sipStringTypeClassMap;


/*
 * Define a mapping between a wrapped type identified by an integer and the
 * corresponding Python type.  This is deprecated.
 */
typedef struct _sipIntTypeClassMap {
    /* The type as an integer. */
    int typeInt;

    /* A pointer to the Python type. */
    struct _sipWrapperType **pyType;
} sipIntTypeClassMap;


/*
 * A Python method's component parts.  This allows us to re-create the method
 * without changing the reference counts of the components.
 */
typedef struct _sipPyMethod {
    /* The function. */
    PyObject *mfunc;

    /* Self if it is a bound method. */
    PyObject *mself;

#if PY_MAJOR_VERSION < 3
    /* The class. */
    PyObject *mclass;
#endif
} sipPyMethod;


/*
 * A slot (in the Qt, rather than Python, sense).
 */
typedef struct _sipSlot {
    /* Name if a Qt or Python signal. */
    char *name;

    /* Signal or Qt slot object. */
    PyObject *pyobj;

    /* Python slot method, pyobj is NULL. */
    sipPyMethod meth;

    /* A weak reference to the slot, Py_True if pyobj has an extra reference. */
    PyObject *weakSlot;
} sipSlot;


/*
 * The API exported by the SIP module, ie. pointers to all the data and
 * functions that can be used by generated code.
 */
typedef struct _sipAPIDef {
    /*
     * This must be the first entry and it's signature must not change so that
     * version number mismatches can be detected and reported.
     */
    int (*api_export_module)(sipExportedModuleDef *client, unsigned api_major,
            unsigned api_minor, void *unused);

    /*
     * The following are part of the public API.
     */
    PyTypeObject *api_simplewrapper_type;
    PyTypeObject *api_wrapper_type;
    PyTypeObject *api_wrappertype_type;
    PyTypeObject *api_voidptr_type;

    void (*api_bad_catcher_result)(PyObject *method);
    void (*api_bad_length_for_slice)(SIP_SSIZE_T seqlen, SIP_SSIZE_T slicelen);
    PyObject *(*api_build_result)(int *isErr, const char *fmt, ...);
    PyObject *(*api_call_method)(int *isErr, PyObject *method, const char *fmt,
            ...);
    PyObject *(*api_connect_rx)(PyObject *txObj, const char *sig,
            PyObject *rxObj, const char *slot, int type);
    SIP_SSIZE_T (*api_convert_from_sequence_index)(SIP_SSIZE_T idx,
            SIP_SSIZE_T len);
    int (*api_can_convert_to_type)(PyObject *pyObj, const sipTypeDef *td,
            int flags);
    void *(*api_convert_to_type)(PyObject *pyObj, const sipTypeDef *td,
            PyObject *transferObj, int flags, int *statep, int *iserrp);
    void *(*api_force_convert_to_type)(PyObject *pyObj, const sipTypeDef *td,
            PyObject *transferObj, int flags, int *statep, int *iserrp);
    int (*api_can_convert_to_enum)(PyObject *pyObj, const sipTypeDef *td);
    void (*api_release_type)(void *cpp, const sipTypeDef *td, int state);
    PyObject *(*api_convert_from_type)(void *cpp, const sipTypeDef *td,
            PyObject *transferObj);
    PyObject *(*api_convert_from_new_type)(void *cpp, const sipTypeDef *td,
            PyObject *transferObj);
    PyObject *(*api_convert_from_enum)(int eval, const sipTypeDef *td);
    int (*api_get_state)(PyObject *transferObj);
    PyObject *(*api_disconnect_rx)(PyObject *txObj, const char *sig,
            PyObject *rxObj, const char *slot);
    void (*api_free)(void *mem);
    PyObject *(*api_get_pyobject)(void *cppPtr, const sipTypeDef *td);
    void *(*api_malloc)(size_t nbytes);
    int (*api_parse_result)(int *isErr, PyObject *method, PyObject *res,
            const char *fmt, ...);
    void (*api_trace)(unsigned mask, const char *fmt, ...);
    void (*api_transfer_back)(PyObject *self);
    void (*api_transfer_to)(PyObject *self, PyObject *owner);
    void (*api_transfer_break)(PyObject *self);
    unsigned long (*api_long_as_unsigned_long)(PyObject *o);
    PyObject *(*api_convert_from_void_ptr)(void *val);
    PyObject *(*api_convert_from_const_void_ptr)(const void *val);
    PyObject *(*api_convert_from_void_ptr_and_size)(void *val,
            SIP_SSIZE_T size);
    PyObject *(*api_convert_from_const_void_ptr_and_size)(const void *val,
            SIP_SSIZE_T size);
    void *(*api_convert_to_void_ptr)(PyObject *obj);
    int (*api_export_symbol)(const char *name, void *sym);
    void *(*api_import_symbol)(const char *name);
    const sipTypeDef *(*api_find_type)(const char *type);
    int (*api_register_py_type)(PyTypeObject *type);
    const sipTypeDef *(*api_type_from_py_type_object)(PyTypeObject *py_type);
    const sipTypeDef *(*api_type_scope)(const sipTypeDef *td);
    const char *(*api_resolve_typedef)(const char *name);
    int (*api_register_attribute_getter)(const sipTypeDef *td,
            sipAttrGetterFunc getter);
    int (*api_is_api_enabled)(const char *name, int from, int to);
    sipErrorState (*api_bad_callable_arg)(int arg_nr, PyObject *arg);
    void *(*api_get_address)(struct _sipSimpleWrapper *w);
    void (*api_set_destroy_on_exit)(int);
    int (*api_enable_autoconversion)(const sipTypeDef *td, int enable);

    /*
     * The following are deprecated parts of the public API.
     */
    PyTypeObject *(*api_find_named_enum)(const char *type);
    const sipMappedType *(*api_find_mapped_type)(const char *type);
    sipWrapperType *(*api_find_class)(const char *type);
    sipWrapperType *(*api_map_int_to_class)(int typeInt,
            const sipIntTypeClassMap *map, int maplen);
    sipWrapperType *(*api_map_string_to_class)(const char *typeString,
            const sipStringTypeClassMap *map, int maplen);

    /*
     * The following may be used by Qt support code but no other handwritten
     * code.
     */
    void (*api_free_sipslot)(sipSlot *slot);
    int (*api_same_slot)(const sipSlot *sp, PyObject *rxObj, const char *slot);
    void *(*api_convert_rx)(sipWrapper *txSelf, const char *sigargs,
            PyObject *rxObj, const char *slot, const char **memberp,
            int flags);
    PyObject *(*api_invoke_slot)(const sipSlot *slot, PyObject *sigargs);
    int (*api_save_slot)(sipSlot *sp, PyObject *rxObj, const char *slot);
    void (*api_clear_any_slot_reference)(sipSlot *slot);
    int (*api_visit_slot)(sipSlot *slot, visitproc visit, void *arg);

    /*
     * The following are not part of the public API.
     */
    int (*api_init_module)(sipExportedModuleDef *client, PyObject *mod_dict);
    int (*api_parse_args)(PyObject **parseErrp, PyObject *sipArgs,
            const char *fmt, ...);
    int (*api_parse_pair)(PyObject **parseErrp, PyObject *arg0, PyObject *arg1,
            const char *fmt, ...);
    void (*api_common_dtor)(sipSimpleWrapper *sipSelf);
    void (*api_no_function)(PyObject *parseErr, const char *func,
            const char *doc);
    void (*api_no_method)(PyObject *parseErr, const char *scope,
            const char *method, const char *doc);
    void (*api_abstract_method)(const char *classname, const char *method);
    void (*api_bad_class)(const char *classname);
    void *(*api_get_cpp_ptr)(sipSimpleWrapper *w, const sipTypeDef *td);
    void *(*api_get_complex_cpp_ptr)(sipSimpleWrapper *w);
    PyObject *(*api_is_py_method)(sip_gilstate_t *gil, char *pymc,
            sipSimpleWrapper *sipSelf, const char *cname, const char *mname);
    void (*api_call_hook)(const char *hookname);
    void (*api_end_thread)(void);
    void (*api_raise_unknown_exception)(void);
    void (*api_raise_type_exception)(const sipTypeDef *td, void *ptr);
    int (*api_add_type_instance)(PyObject *dict, const char *name,
            void *cppPtr, const sipTypeDef *td);
    void (*api_bad_operator_arg)(PyObject *self, PyObject *arg,
            sipPySlotType st);
    PyObject *(*api_pyslot_extend)(sipExportedModuleDef *mod, sipPySlotType st,
            const sipTypeDef *type, PyObject *arg0, PyObject *arg1);
    void (*api_add_delayed_dtor)(sipSimpleWrapper *w);
    char (*api_bytes_as_char)(PyObject *obj);
    const char *(*api_bytes_as_string)(PyObject *obj);
    char (*api_string_as_ascii_char)(PyObject *obj);
    const char *(*api_string_as_ascii_string)(PyObject **obj);
    char (*api_string_as_latin1_char)(PyObject *obj);
    const char *(*api_string_as_latin1_string)(PyObject **obj);
    char (*api_string_as_utf8_char)(PyObject *obj);
    const char *(*api_string_as_utf8_string)(PyObject **obj);
#if defined(HAVE_WCHAR_H)
    wchar_t (*api_unicode_as_wchar)(PyObject *obj);
    wchar_t *(*api_unicode_as_wstring)(PyObject *obj);
#else
    int (*api_unicode_as_wchar)(PyObject *obj);
    int *(*api_unicode_as_wstring)(PyObject *obj);
#endif
    int (*api_deprecated)(const char *classname, const char *method);
    void (*api_keep_reference)(PyObject *self, int key, PyObject *obj);
    int (*api_parse_kwd_args)(PyObject **parseErrp, PyObject *sipArgs,
            PyObject *sipKwdArgs, const char **kwdlist, PyObject **unused,
            const char *fmt, ...);
    void (*api_add_exception)(sipErrorState es, PyObject **parseErrp);
    int (*api_parse_result_ex)(sip_gilstate_t, sipVirtErrorHandlerFunc,
            sipSimpleWrapper *, PyObject *method, PyObject *res,
            const char *fmt, ...);
    void (*api_call_error_handler)(sipVirtErrorHandlerFunc,
            sipSimpleWrapper *, sip_gilstate_t);
    int (*api_init_mixin)(PyObject *self, PyObject *args, PyObject *kwds,
            const sipClassTypeDef *ctd);
    /*
     * The following are part of the public API.
     */
    void *(*api_get_mixin_address)(struct _sipSimpleWrapper *w,
            const sipTypeDef *td);
    PyObject *(*api_convert_from_new_pytype)(void *cpp, PyTypeObject *py_type,
            sipWrapper *owner, sipSimpleWrapper **selfp, const char *fmt, ...);
    PyObject *(*api_convert_to_typed_array)(void *data, const sipTypeDef *td,
            const char *format, size_t stride, SIP_SSIZE_T len, int flags);
    PyObject *(*api_convert_to_array)(void *data, const char *format,
            SIP_SSIZE_T len, int flags);
    int (*api_register_proxy_resolver)(const sipTypeDef *td,
            sipProxyResolverFunc resolver);

    /*
     * The following may be used by Qt support code but no other handwritten
     * code.
     */
    PyObject *(*api_invoke_slot_ex)(const sipSlot *slot, PyObject *sigargs,
            int check_receiver);

    /*
     * The following is not part of the public API.
     */
    PyObject *(*api_get_reference)(PyObject *self, int key);

    /*
     * The following is part of the public API.
     */
    PyInterpreterState *(*api_get_interpreter)();
} sipAPIDef;


/*
 * The API implementing the optional Qt support.
 */
typedef struct _sipQtAPI {
    sipTypeDef **qt_qobject;
    void *(*qt_create_universal_signal)(void *, const char **);
    void *(*qt_find_universal_signal)(void *, const char **);
    void *(*qt_create_universal_slot)(struct _sipWrapper *, const char *,
            PyObject *, const char *, const char **, int);
    void (*qt_destroy_universal_slot)(void *);
    void *(*qt_find_slot)(void *, const char *, PyObject *, const char *,
            const char **);
    int (*qt_connect)(void *, const char *, void *, const char *, int);
    int (*qt_disconnect)(void *, const char *, void *, const char *);
    int (*qt_same_name)(const char *, const char *);
    sipSlot *(*qt_find_sipslot)(void *, void **);
    int (*qt_emit_signal)(PyObject *, const char *, PyObject *);
    int (*qt_connect_py_signal)(PyObject *, const char *, PyObject *,
            const char *);
    void (*qt_disconnect_py_signal)(PyObject *, const char *, PyObject *,
            const char *);
} sipQtAPI;


/*
 * These are flags that can be passed to sipCanConvertToType(),
 * sipConvertToType() and sipForceConvertToType().
 */
#define SIP_NOT_NONE        0x01    /* Disallow None. */
#define SIP_NO_CONVERTORS   0x02    /* Disable any type convertors. */


/*
 * These are flags that can be passed to sipConvertToArray().
 */
#define SIP_READ_ONLY       0x01    /* The array is read-only. */
#define SIP_OWNS_MEMORY     0x02    /* The array owns its memory. */


/*
 * These are the state flags returned by %ConvertToTypeCode.  Note that these
 * share the same "namespace" as the flags below.
 */
#define SIP_TEMPORARY       0x0001  /* A temporary instance. */
#define SIP_DERIVED_CLASS   0x0002  /* The instance is derived. */


/*
 * These flags are specific to the Qt support API.
 */
#define SIP_SINGLE_SHOT     0x01    /* The connection is single shot. */


/*
 * Useful macros, not part of the public API.
 */
#define SIP_PY_OWNED        0x0004  /* If owned by Python. */
#define SIP_INDIRECT        0x0008  /* If there is a level of indirection. */
#define SIP_ACCFUNC         0x0010  /* If there is an access function. */
#define SIP_NOT_IN_MAP      0x0020  /* If Python object is not in the map. */
#define SIP_SHARE_MAP       0x0040  /* If the map slot might be occupied. */
#define SIP_CPP_HAS_REF     0x0080  /* If C/C++ has a reference. */
#define SIP_POSSIBLE_PROXY  0x0100  /* If there might be a proxy slot. */
#define SIP_ALIAS           0x0200  /* If it is an alias. */
#define SIP_CREATED         0x0400  /* If the C/C++ object has been created. */

#define sipIsPyOwned(w)     ((w)->flags & SIP_PY_OWNED)
#define sipSetPyOwned(w)    ((w)->flags |= SIP_PY_OWNED)
#define sipResetPyOwned(w)  ((w)->flags &= ~SIP_PY_OWNED)
#define sipIsDerived(w)     ((w)->flags & SIP_DERIVED_CLASS)
#define sipIsIndirect(w)    ((w)->flags & SIP_INDIRECT)
#define sipIsAccessFunc(w)  ((w)->flags & SIP_ACCFUNC)
#define sipNotInMap(w)      ((w)->flags & SIP_NOT_IN_MAP)
#define sipSetNotInMap(w)   ((w)->flags |= SIP_NOT_IN_MAP)
#define sipCppHasRef(w)     ((w)->flags & SIP_CPP_HAS_REF)
#define sipSetCppHasRef(w)  ((w)->flags |= SIP_CPP_HAS_REF)
#define sipResetCppHasRef(w)    ((w)->flags &= ~SIP_CPP_HAS_REF)
#define sipPossibleProxy(w) ((w)->flags & SIP_POSSIBLE_PROXY)
#define sipSetPossibleProxy(w)  ((w)->flags |= SIP_POSSIBLE_PROXY)
#define sipIsAlias(w)       ((w)->flags & SIP_ALIAS)
#define sipWasCreated(w)    ((w)->flags & SIP_CREATED)


#define SIP_TYPE_TYPE_MASK  0x0007  /* The type type mask. */
#define SIP_TYPE_CLASS      0x0000  /* If the type is a C++ class. */
#define SIP_TYPE_NAMESPACE  0x0001  /* If the type is a C++ namespace. */
#define SIP_TYPE_MAPPED     0x0002  /* If the type is a mapped type. */
#define SIP_TYPE_ENUM       0x0003  /* If the type is a named enum. */
#define SIP_TYPE_ABSTRACT   0x0008  /* If the type is abstract. */
#define SIP_TYPE_SCC        0x0010  /* If the type is subject to sub-class convertors. */
#define SIP_TYPE_ALLOW_NONE 0x0020  /* If the type can handle None. */
#define SIP_TYPE_STUB       0x0040  /* If the type is a stub. */
#define SIP_TYPE_NONLAZY    0x0080  /* If the type has a non-lazy method. */
#define SIP_TYPE_SUPER_INIT 0x0100  /* If the instance's super init should be called. */


/*
 * The following are part of the public API.
 */
#define sipTypeIsClass(td)  (((td)->td_flags & SIP_TYPE_TYPE_MASK) == SIP_TYPE_CLASS)
#define sipTypeIsNamespace(td)  (((td)->td_flags & SIP_TYPE_TYPE_MASK) == SIP_TYPE_NAMESPACE)
#define sipTypeIsMapped(td) (((td)->td_flags & SIP_TYPE_TYPE_MASK) == SIP_TYPE_MAPPED)
#define sipTypeIsEnum(td)   (((td)->td_flags & SIP_TYPE_TYPE_MASK) == SIP_TYPE_ENUM)
#define sipTypeAsPyTypeObject(td)   ((td)->u.td_py_type)
#define sipTypeName(td)     sipNameFromPool((td)->td_module, (td)->td_cname)

#define sipIsExactWrappedType(wt)   (sipTypeAsPyTypeObject((wt)->type) == (PyTypeObject *)(wt))

#if PY_VERSION_HEX >= 0x03020000
#define sipConvertFromSliceObject   PySlice_GetIndicesEx
#else
#define sipConvertFromSliceObject(o, len, start, stop, step, slen) \
        PySlice_GetIndicesEx((PySliceObject *)(o), (len), (start), (stop), \
                (step), (slen))
#endif


/*
 * The following are deprecated parts of the public API.
 */
#define sipClassName(w)     PyString_FromString(Py_TYPE(w)->tp_name)


/*
 * The following are not part of the public API.
 */
#define sipTypeIsAbstract(td)   ((td)->td_flags & SIP_TYPE_ABSTRACT)
#define sipTypeHasSCC(td)   ((td)->td_flags & SIP_TYPE_SCC)
#define sipTypeAllowNone(td)    ((td)->td_flags & SIP_TYPE_ALLOW_NONE)
#define sipTypeIsStub(td)   ((td)->td_flags & SIP_TYPE_STUB)
#define sipTypeSetStub(td)  ((td)->td_flags |= SIP_TYPE_STUB)
#define sipTypeHasNonlazyMethod(td) ((td)->td_flags & SIP_TYPE_NONLAZY)
#define sipTypeCallSuperInit(td)    ((td)->td_flags & SIP_TYPE_SUPER_INIT)

/*
 * Get various names from the string pool for various data types.
 */
#define sipNameFromPool(em, mr) (&((em)->em_strings)[(mr)])
#define sipNameOfModule(em)     sipNameFromPool((em), (em)->em_name)
#define sipPyNameOfContainer(cod, td)   sipNameFromPool((td)->td_module, (cod)->cod_name)
#define sipPyNameOfEnum(etd)    sipNameFromPool((etd)->etd_base.td_module, (etd)->etd_name)


/*
 * The following are PyQt3-specific extensions.  In SIP v5 they will be pushed
 * out to a plugin supplied by PyQt3.
 */


/*
 * Maps the name of a Qt signal to a wrapper function to emit it.
 */
typedef int (*pyqt3EmitFunc)(sipSimpleWrapper *, PyObject *);

typedef struct _pyqt3QtSignal {
    /* The signal name. */
    const char *st_name;

    /* The emitter function. */
    pyqt3EmitFunc st_emitfunc;
} pyqt3QtSignal;


/*
 * This is the PyQt3-specific extension to the generated class type structure.
 */
typedef struct _pyqt3ClassTypeDef {
    /*
     * The super-type structure.  This must be first in the structure so that
     * it can be cast to sipClassTypeDef *.
     */
    sipClassTypeDef super;

    /* The emit table for Qt signals. */
    pyqt3QtSignal *qt3_emit;
} pyqt3ClassTypeDef;


/*
 * The following are PyQt4-specific extensions.  In SIP v5 they will be pushed
 * out to a plugin supplied by PyQt4.
 */

/*
 * The description of a Qt signal for PyQt4.
 */
typedef struct _pyqt4QtSignal {
    /* The C++ name and signature of the signal. */
    const char *signature;

    /* The optional docstring. */
    const char *docstring;

    /*
     * If the signal is an overload of regular methods then this points to the
     * code that implements those methods.
     */
    PyMethodDef *non_signals;

    /*
     * The hack to apply when built against Qt5:
     *
     * 0 - no hack
     * 1 - add an optional None
     * 2 - add an optional []
     * 3 - add an optional False
     */
    int hack;
} pyqt4QtSignal;


/*
 * This is the PyQt4-specific extension to the generated class type structure.
 */
typedef struct _pyqt4ClassTypeDef {
    /*
     * The super-type structure.  This must be first in the structure so that
     * it can be cast to sipClassTypeDef *.
     */
    sipClassTypeDef super;

    /* A pointer to the QObject sub-class's staticMetaObject class variable. */
    const void *static_metaobject;

    /*
     * A set of flags.  At the moment only bit 0 is used to say if the type is
     * derived from QFlags.
     */
    unsigned flags;

    /*
     * The table of signals emitted by the type.  These are grouped by signal
     * name.
     */
    const pyqt4QtSignal *qt_signals;
} pyqt4ClassTypeDef;


/*
 * The following are PyQt5-specific extensions.  In SIP v5 they will be pushed
 * out to a plugin supplied by PyQt5.
 */

/*
 * The description of a Qt signal for PyQt5.
 */
typedef int (*pyqt5EmitFunc)(void *, PyObject *);

typedef struct _pyqt5QtSignal {
    /* The normalised C++ name and signature of the signal. */
    const char *signature;

    /* The optional docstring. */
    const char *docstring;

    /*
     * If the signal is an overload of regular methods then this points to the
     * code that implements those methods.
     */
    PyMethodDef *non_signals;

    /*
     * If the signal has optional arguments then this function will implement
     * emit() for the signal.
     */
    pyqt5EmitFunc emitter;
} pyqt5QtSignal;


/*
 * This is the PyQt5-specific extension to the generated class type structure.
 */
typedef struct _pyqt5ClassTypeDef {
    /*
     * The super-type structure.  This must be first in the structure so that
     * it can be cast to sipClassTypeDef *.
     */
    sipClassTypeDef super;

    /* A pointer to the QObject sub-class's staticMetaObject class variable. */
    const void *static_metaobject;

    /*
     * A set of flags.  At the moment only bit 0 is used to say if the type is
     * derived from QFlags.
     */
    unsigned flags;

    /*
     * The table of signals emitted by the type.  These are grouped by signal
     * name.
     */
    const pyqt5QtSignal *qt_signals;

    /* The name of the interface that the class defines. */
    const char *qt_interface;
} pyqt5ClassTypeDef;


#ifdef __cplusplus
}
#endif


#endif
