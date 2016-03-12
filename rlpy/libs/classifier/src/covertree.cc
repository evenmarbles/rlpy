#include "covertree.h"
#include "array_helper.h"


DistanceFunction::DistanceFunction(const StateVariables &scale) :
m_scale(scale)
{}

double DistanceFunction::operator()(const StateVector &x,
	const StateVector &y) const
{
	double dd = 0.0;
	StateVariables::const_iterator i;
	for (i = m_scale.begin(); i != m_scale.end(); ++i) {
		const unsigned index = i->first;
		const double scale = i->second;
		const double d = scale * ((y)[index] - (x)[index]);
		dd += d*d;
	}
	return std::sqrt(dd);
}


/*
* PyCoverTree_Type
*/


void PyCoverTree_dealloc(PyCoverTreeObject* self)
{
	delete self->tree;
	self->tree = NULL;

	self->ob_type->tp_free((PyObject*)self);
}

PyObject * PyCoverTree_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	PyCoverTreeObject *self;
	self = (PyCoverTreeObject *)type->tp_alloc(type, 0);
	if (self != NULL) {
	}

	return (PyObject *)self;
}

std::vector<double> create_state_vector(PyArrayObject *pyo)
{
	int len;

	len = PyArray_DIM(pyo, 0);
	double * o = pyvector_to_Carrayptrs(pyo, 1, len);

	std::vector<double> result;
	result.insert(result.end(), o, o + len);
	return result;
}

int PyCoverTree_init(PyCoverTreeObject *self, PyObject *args, PyObject *kwds)
{
	int len;
	PyArrayObject *pyscale;

	static char *kwlist[] = { "scale", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyArray_Type, &pyscale))
		return -1;

	if (!PyArray_Check(pyscale) && !PyFloat_Check(pyscale)) {
		return NULL;
	}

	len = PyArray_DIM(pyscale, 0);
	double * scale = pyvector_to_Carrayptrs(pyscale, 1, len);

	StateVariables dimensions;
	for (unsigned i = 0; i < len; ++i) {
		dimensions[i] = scale[i];
	}

	self->tree = new CoverTree<StateVector, DistanceFunction>(DistanceFunction(dimensions));

	return 0;
}

PyObject * PyCoverTree_insert(PyCoverTreeObject * self, PyObject *args)
{
	int dims[1];
	double * new_state;

	PyArrayObject *pyo, *pynewstate;

	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &pyo)) {
		return NULL;
	}

	StateVector state = create_state_vector(pyo);
	StateVector new_state_vec = self->tree->insert(state);

	dims[0] = new_state_vec.size();
	pynewstate = (PyArrayObject *)PyArray_FromDims(1, dims, NPY_DOUBLE);
	if (!pynewstate) return NULL;
	new_state = pyvector_to_Carrayptrs(pynewstate, 1, 1);
	for (int j = 0; j < new_state_vec.size(); ++j)
	{
		new_state[j] = new_state_vec[j];
	}

	return PyArray_Return(pynewstate);
}

PyObject * PyCoverTree_remove(PyCoverTreeObject * self, PyObject *args)
{
	PyArrayObject *pyo;

	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &pyo)) {
		return NULL;
	}

	StateVector state = create_state_vector(pyo);
	bool retval = self->tree->remove(state);

	return PyBool_FromLong(retval);
}

PyObject * PyCoverTree_clear(PyCoverTree * self)
{
	self->tree->clear();
	return Py_None;
}

PyObject * PyCoverTree_neighbors(PyCoverTreeObject * self, PyObject *args)
{
	int dims[1];
	double * state;

	PyArrayObject *pyo, *pystate;
	double maxd;

	if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &pyo, &maxd)) {
		return NULL;
	}

	StateVector basis = create_state_vector(pyo);
	std::vector<std::pair<double, StateVector> > buffer;
	self->tree->neighbors(std::back_inserter(buffer), basis, maxd);

	PyObject * pyneighbors = PyList_New(0);
	std::vector<std::pair<double, StateVector> >::iterator i;
	for (i = buffer.begin(); i != buffer.end(); ++i)
	{
		dims[0] = i->second.size();
		pystate = (PyArrayObject *)PyArray_FromDims(1, dims, NPY_DOUBLE);
		if (!pystate) return NULL;
		state = pyvector_to_Carrayptrs(pystate, 1, 1);
		for (int j = 0; j < i->second.size(); ++j)
		{
			state[j] = i->second[j];
		}
		PyObject * val = PyTuple_Pack(2, PyFloat_FromDouble(i->first), (PyObject *)pystate);
		PyList_Append(pyneighbors, val);
	}
	return pyneighbors;
}

PyObject * PyCoverTree_nearest(PyCoverTreeObject * self, PyObject *args)
{
	int dims[1];
	double * state;

	PyArrayObject *pyo, *pystate;
	unsigned k;

	if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &pyo, &k)) {
		return NULL;
	}

	StateVector basis = create_state_vector(pyo);
	std::vector<std::pair<double, StateVector> > buffer;
	self->tree->nearest(std::back_inserter(buffer), basis, k);

	PyObject * pyneighbors = PyList_New(0);
	std::vector<std::pair<double, StateVector> >::iterator i;
	for (i = buffer.begin(); i != buffer.end(); ++i)
	{
		dims[0] = i->second.size();
		pystate = (PyArrayObject *)PyArray_FromDims(1, dims, NPY_DOUBLE);
		if (!pystate) return NULL;
		state = pyvector_to_Carrayptrs(pystate, 1, 1);
		for (int j = 0; j < i->second.size(); ++j)
		{
			state[j] = i->second[j];
		}

		PyObject * val = PyTuple_Pack(2, PyFloat_FromDouble(i->first), (PyObject *)pystate);
		PyList_Append(pyneighbors, val);
	}
	return pyneighbors;
}

static PyMethodDef PyCoverTree_methods[] = {
	{ "insert", (PyCFunction)PyCoverTree_insert, METH_VARARGS,
	"nserts a new point into the cover tree." },
	{ "remove", (PyCFunction)PyCoverTree_remove, METH_VARARGS,
	"Removes one point, if it exists." },
	{ "clear", (PyCFunction)PyCoverTree_clear, METH_NOARGS,
	"Remove all instances from the tree." },
	{ "neighbors", (PyCFunction)PyCoverTree_neighbors, METH_VARARGS,
	"Find all instances within a certain radius." },
	{ "nearest", (PyCFunction)PyCoverTree_nearest, METH_VARARGS,
	"Find the k nearest neighbors (and their distances)." },
	{ NULL }  /* Sentinel */
};


PyTypeObject PyCoverTree_Type = {
	PyObject_HEAD_INIT(NULL)
	0,                               /*ob_size*/
	"PyCoverTree",					 /*tp_name*/
	sizeof(PyCoverTreeObject),       /*tp_basicsize*/
	0,                               /*tp_itemsize*/
	(destructor)PyCoverTree_dealloc, /*tp_dealloc*/
	0,                               /*tp_print*/
	0,                               /*tp_getattr*/
	0,                               /*tp_setattr*/
	0,                               /*tp_compare*/
	0,                               /*tp_repr*/
	0,                               /*tp_as_number*/
	0,                               /*tp_as_sequence*/
	0,                               /*tp_as_mapping*/
	0,                               /*tp_hash */
	0,                               /*tp_call*/
	0,                               /*tp_str*/
	0,                               /*tp_getattro*/
	0,                               /*tp_setattro*/
	0,                               /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
	"PyC45Tree objects",             /* tp_doc */
	0,		                         /* tp_traverse */
	0,		                         /* tp_clear */
	0,		                         /* tp_richcompare */
	0,		                         /* tp_weaklistoffset */
	0,		                         /* tp_iter */
	0,		                         /* tp_iternext */
	PyCoverTree_methods,             /* tp_methods */
	0,								 /* tp_members */
	0,								 /* tp_getset */
	0,                               /* tp_base */
	0,                               /* tp_dict */
	0,                               /* tp_descr_get */
	0,                               /* tp_descr_set */
	0,                               /* tp_dictoffset */
	(initproc)PyCoverTree_init,      /* tp_init */
	0,                               /* tp_alloc */
	PyCoverTree_new,                 /* tp_new */
};
