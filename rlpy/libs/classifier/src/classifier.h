#ifndef CLASSIFIER_H
#define CLASSIFIER_H

# ifdef _DEBUG
// Include these low level headers before undefing _DEBUG. Otherwise when doing
// a debug build against a release build of python the compiler will end up 
// including these low level headers without DEBUG enabled, causing it to try 
// and link release versions of this low level C api.
# include <basetsd.h>
# include <assert.h>
# include <ctype.h>
# include <errno.h>
# include <io.h>
# include <math.h>
# include <sal.h>
# include <stdarg.h>
# include <stddef.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <sys/stat.h>
# include <time.h>
# include <wchar.h>
#  undef _DEBUG
#  if defined(_MSC_VER) && _MSC_VER >= 1400
#    define _CRT_NOFORCE_MANIFEST 1
#  endif
#  include <Python.h>
#  define _DEBUG
# else
#  include <Python.h>
# endif

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_Classifier
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"


/*
* PyClassPair_Type
*/
typedef struct _classpairobject PyClassPairObject;
struct _classpairobject {
	PyObject_HEAD
	PyArrayObject *in_; /* tree input */
	double out;	/* tree output */
};

extern PyTypeObject PyClassPair_Type;

#define PyClassPair_Check(op) PyObject_TypeCheck(op, &PyClassPair_Type)
#define PyClassPair_CheckExact(op) (Py_TYPE(op) == &PyClassPair_Type)



/*
* PyClassPairList_Type
*/
typedef struct _classpairlistobject PyClassPairListObject;
struct _classpairlistobject {
	PyListObject list;
};

extern PyTypeObject PyClassPairList_Type;

#define PyClassPairList_Check(op) PyObject_TypeCheck(op, &PyClassPairList_Type)
#define PyClassPairList_CheckExact(op) (Py_TYPE(op) == &PyClassPairList_Type)

#endif	// CLASSIFIER_H
