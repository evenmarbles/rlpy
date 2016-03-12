/* Header for C4.5 decision tree algorithm for Python: c45module.cpp */

/* ==== Prototypes =================================== */

#ifndef CLASSIFIER_MODULE_HH
#define CLASSIFIER_MODULE_HH

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _DEBUG
#define _DEBUG_WAS_DEFINED 1
#undef _DEBUG
#endif

#include "Python.h"
#include "structmember.h"

#ifdef _DEBUG_WAS_DEFINED
#define _DEBUG 1
#endif

#if !defined(CLASSIFIERMODULE_IMPORT_ARRAY)
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_Classifier
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"

#ifdef __cplusplus
}
#endif

#endif		// CLASSIFIER_MODULE_HH