#ifndef _COVERTREE_HH_
#define _COVERTREE_HH_

/** \file
	Implementation of the Cover Tree algorithm.
	http://hunch.net/~jl/projects/cover_tree/cover_tree.html */

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

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <utility>
#include <vector>

/** Type representing an individual state.  The agent code uses this
type as its internal representation of a state.  Therefore, it
first transforms any observation from RL Glue into a
std::vector<double>.  Smart pointers allow us to keep in memory
only those instances still used by the agent. */
typedef std::vector<double> StateVector;

/** A data structure that specifies how to interpret state vectors by
weighting the state variables.  Each key is an index into a state
vector, and the corresponding datum is the (positive) factor by
which to multiply that state variable's value, to obtain the
normalized state.  Straight Euclidean distance in this transformed
space emulates scaled Euclidean distance in the original space.
State vector indices that do not appear in the map are assumed to
be irrelevant: the data associated with these keys are assumed to
be 0, in effect implementing state abstraction. */
typedef std::map<size_t, double> StateVariables;

/** Scaled Euclidean distance between states.  This implementation may
optionally ignore certain dimensions of the state space.*/
class DistanceFunction :
	public std::binary_function<StateVector, StateVector, double>
{
public:
	/** \param scale Determines how to scale the inputs before computing
	distances. */
	DistanceFunction(const StateVariables &scale);

	/** Computes scaled Euclidean distance between two states.
	\param x State with size() larger than the greatest index in the
	scale mapping provided to this Distance Function.
	\param y State with size() larger than the greatest index in the
	scale mapping provided to this Distance Function.
	\return The scaled Euclidean distance in the specified
	dimensions */
	double operator()(const StateVector &x, const StateVector &y) const;

private:
	const StateVariables m_scale;
};

/** Container for data of type T that supports searches for nearest
	neighbors, using a distance function object of type Metric. */
template <class T, class Metric>
class CoverTree {
protected:
	/** Representation of an individual instance. */
	struct Node {
		/** Each child node has an associated level at which it first
		appears. */
		typedef std::pair<int, Node> Child;

		/** Each node can have an unbounded number of children. */
		typedef std::vector<Child> ChildContainer;

		/** Comparator for sorting the children by depth. */
		struct Compare : public std::binary_function<Child, Child, bool> {
			bool operator()(const Child &a, const Child &b) const {
				return a.first > b.first;
			}
		};

		/** Construct a new Node with no children.
			\param x The instance the node will contain. */
		Node(const T &x) : x(x) {}

		T x;
		ChildContainer children;
	};

	/** Object that tracks the iteration through a Node's list of
		children.  Some cover tree operations must iterate through
		several Node's children simultaneously. */
	struct LinkData {
		/** Comparator that compares LinkData objects by the level of each
		one's current Node.  The current node at the front of a
		priority queue using this comparator will have the highest
		level among all nodes in the priority queue. */
		struct Compare : public std::binary_function<LinkData, LinkData, bool> {
			bool operator()(const LinkData &a, const LinkData &b) const {
				return a.level() < b.level();
			}
		};

		/** Construct a new LinkData iteration.
		\param node The node through whose children to iterate.
		\param d A distance from node->x that governs how long the
		iteration is relevant to a particular query or
		operation. */
		LinkData(Node *node, double d) :
			it(node->children.begin()), end(node->children.end()), d(d) {}

		/** The first level at which the current node in this iteration
		appears. */
		int level() const {
			return it->first;
		}

		/** Whether any of the remaining nodes in this iteration is
		relevant to an instance at distance (d + margin) from the node
		that originated this LinkData. */
		bool relevant(double margin = 0) const {
			return it != end && d < margin + std::ldexp(1., level() + 1);
		}

		typename Node::ChildContainer::iterator it;
		typename Node::ChildContainer::iterator end;
		double d;
	};

public:
	/** \param duplicate Whether to add an instance that is at distance
						 0 from an existing instance. */
	CoverTree(bool duplicate = false) :
		root(NULL), duplicate(duplicate)
	{}

	/** \param d The distance metric to use.
		\param duplicate Whether to add an instance that is at distance
		0 from an existing instance. */
	CoverTree(const Metric &d, bool duplicate = false) :
		root(NULL), d(d), duplicate(duplicate)
	{}

	/** Copy constructor.  Assumes that T and Metric are copyable. */
	CoverTree(const CoverTree &other) :
		root(new Node(*other.root)),
		d(other.d),
		duplicate(other.duplicate)
	{}

	~CoverTree() {
		delete root;
	}

	/** Assignment operator.  Assumes that T and Metric are assignable. */
	CoverTree &operator=(const CoverTree &other) {
		delete root;
		root = new Node(other->root);
		d = other.d;
		duplicate = other.duplicate;
	}

	/** Inserts a new point into the cover tree
		\param x The point to insert
		\return The inserted instance, or if this CoverTree does not
		insert duplicates, an existing instance at distance 0
		from x */
	const T &insert(const T &x) {
		return insert_node(x)->x;
	}

	/** Removes one point, if it exists.  Assumes that T can be tested
		for equality.
		\return True if a point was removed */
	bool remove(const T &x) {
		if (root == NULL)
			return false;
		if (root->x == x) {
			Node *old = root;
			root = NULL;
			move_children(*old); // Reinsert all descendents into tree.
			delete old;
			return true;
		}
		return remove_recursive(x, *root);
	}

	/** Find the k nearest neighbors (and their distances).
		\param out An output iterator into which to write the results.
		The value_type of this iterator must be
		std::pair<double, T>.
		\param x The instance whose neighbors to find
		\param k The number of neighbors to find.
		\return The output iterator incremented (and written) up to k
		times.  The nearest instance will be written first. */
	template <class OutputIterator>
	OutputIterator nearest(OutputIterator out, const T &x, unsigned k) {
		if (root != NULL) {
			// The k nearest instances found so far, initialized to the
			// root, along with their distances from x.
			std::vector<std::pair<double, Node *> > nearest;
			nearest.reserve(k);
			nearest.push_back(std::make_pair(d(x, root->x), root));

			// A priority queue of iterators through all the nodes relevant
			// to this search, initialized to the first child of the root.
			std::vector<LinkData> pqueue;
			LinkData front = LinkData(nearest.front().second, nearest.front().first);
			if (front.it != front.end) {
				pqueue.push_back(front);
			}
			typename LinkData::Compare cmp;

			// Traverse the cover tree by order of level, pruning branches
			// that are too far from x to be a nearest neighbor.
			while (!pqueue.empty()) {
				const int level = pqueue.front().level();

				// If a node is farther away from x than this threshold, then
				// none of its children can be one of the k nearest neighbors.
				const double threshold =
					nearest.back().first + std::ldexp(1., level + 1);

				// Search one entire level at a time.
				while (!pqueue.empty() && pqueue.front().level() == level) {
					front = pqueue.front();
					std::pop_heap(pqueue.begin(), pqueue.end(), cmp);
					pqueue.pop_back();

					// Look at all of this LinkData's children that joined the
					// tree at this level.
					while (front.it != front.end && front.level() == level) {
						if (front.d < threshold) {
							// The current child may be a nearest neighbor or an
							// ancestor of one.
							Node *node = &front.it->second;
							const double dist = d(x, node->x);

							// If this node is nearer than the last node in nearest (or
							// if nearest is not yet at k instances), then add it to
							// our results.
							if (dist < nearest.back().first || nearest.size() < k) {
								while (nearest.size() >= k)
									nearest.pop_back();
								std::pair<double, Node *> val = std::make_pair(dist, node);
								nearest.insert(std::lower_bound(nearest.begin(),
									nearest.end(),
									val),
									val);
							}

							// Try adding the children of this node to the queue.
							LinkData child = LinkData(node, dist);
							if (child.relevant(nearest.back().first)) {
								pqueue.push_back(child);
								std::push_heap(pqueue.begin(), pqueue.end(), cmp);
							}
						}
						++front.it;
					}

					// The level of the next child in this LinkData has
					// decreased.  See if it's still relevant to our search.
					if (front.relevant(nearest.back().first)) {
						pqueue.push_back(front);
						std::push_heap(pqueue.begin(), pqueue.end(), cmp);
					}
				}
			}

			// Output the results.
			typename std::vector<std::pair<double, Node *> >::const_iterator it;
			for (it = nearest.begin(); it != nearest.end(); ++it)
				*out++ = std::make_pair(it->first, it->second->x);
		}
		return out;
	}

	/** Find all instances within a certain radius.
		\param out An output iterator into which to write the results.
		The value_type of this iterator must be
		std::pair<double, T>.
		\param x The instance whose neighbors to find
		\param margin The search radius.
		\return The output iterator incremented (and written) once for
		each instance with the search radius.  The instances are
		not written in any guaranteed order.  */
	template <class OutputIterator>
	OutputIterator neighbors(OutputIterator out,
		const T &x,
		double margin) const
	{
		if (root != NULL)
			out = recursive_neighbors(out, x, margin, *root);
		return out;
	}

	/** Output all the instances in the cover tree.
		\param out An output iterator into which to write the results.
		The value_type of this iterator must be T. */
	template <class OutputIterator>
	OutputIterator contents(OutputIterator out) const {
		return recursive_contents(out, *root);
	}

	/** Write the tree structure to an output stream.  Each line of the
		output will contain three tab-separated entries, describing a
		link in the tree.  The first entry is the level at which the
		link occurs.  The second and third entries give the parent and
		child instance values, respectively.  This method therefore
		outputs n - 1 lines, where n is the number of instances.  It
		assumes that operator<< is defined for T. */
	void print(std::ostream &out) const {
		if (root != NULL) {
			print(out, root);
		}
	}

	/** Remove all instances from the tree. */
	void clear() {
		delete root;
		root = NULL;
	}

protected:
	/** Inserts a new point into the cover tree
		\param x The point to insert */
	Node *insert_node(const T &x) {
		Node *retval;
		if (root == NULL) {
			root = new Node(x);
			retval = root;
		}
		else {
			// Keep track of closest valid parent instance for x, as well as
			// its distance from x.
			double min = d(x, root->x);
			Node *parent = root;

			// Search the tree by iterating through each node's children, in
			// order of level, while each child list contains instances that
			// may be the closest node (and are therefore relevant).
			std::vector<LinkData> pqueue; // x.relevant() for all x in pqueue
			const LinkData rootdata(parent, min);
			int level;
			if (rootdata.relevant()) {
				level = rootdata.level();
				pqueue.push_back(rootdata);
			}
			typename LinkData::Compare cmp;

			// Look for the first level such that x is very far from every
			// point at that level, while also tracking the closest covering
			// point
			while (!pqueue.empty() && min < std::ldexp(1., level + 1) && min > 0) {
				level = pqueue.front().level();
				// Search one full level at a time.
				while (!pqueue.empty() && pqueue.front().level() == level) {
					LinkData current = pqueue.front();
					std::pop_heap(pqueue.begin(), pqueue.end(), cmp);
					pqueue.pop_back();
					// Check all children in this LinkData that join the tree at
					// this level.
					while (current.it != current.end && current.level() == level) {
						Node *node = &current.it->second;
						const double dist = d(x, node->x);
						if (min > dist && dist < std::ldexp(1., level - 1)) {
							// This node is now the closest valid parent so far.
							min = dist;
							parent = node;
						}

						// Add this node's children to the queue, if relevant.
						LinkData child(node, dist);
						if (child.relevant()) {
							pqueue.push_back(child);
							std::push_heap(pqueue.begin(), pqueue.end(), cmp);
						}
						++current.it;
					}

					// Add the remaining children of this LinkData back to the
					// queue, if relevant.
					if (current.relevant()) {
						pqueue.push_back(current);
						std::push_heap(pqueue.begin(), pqueue.end(), cmp);
					}
				}
			}

			if (min > 0 || duplicate) {
				// Insert x if it has positive distance from all existing
				// instances or if we allow duplicates.
				if (min > 0)
					std::frexp(min, &level); // Level at which x joins the tree
				else
					level = std::numeric_limits<int>::min(); // For duplicates
				typename Node::Child val(level, Node(x));
				typename Node::ChildContainer::iterator pos =
					std::lower_bound(parent->children.begin(), parent->children.end(),
					val, typename Node::Compare());
				pos = parent->children.insert(pos, val);
				retval = &pos->second;
			}
			else
				retval = parent;
		}
		return retval;
	}

	/** Add back to the tree the children of a just-deleted node. */
	void move_children(Node &parent) {
		typename Node::ChildContainer::iterator it = parent.children.begin();
		while (it != parent.children.end()) {
			Node &child = it->second;
			insert_node(child.x);
			move_children(child);
			++it;
		}
	}

	/** Remove an instance from a subtree.
		\param x The instance to remove
		\param node The root of the subtree to search.
		\return Whether x was a descendant of node (and was removed) */
	bool remove_recursive(const T &x, Node &node) {
		const double dist = d(x, node.x);
		typename Node::ChildContainer::iterator it = node.children.begin();

		if (dist == 0) {
			// We assume node.x != x, so x must be a direct child of this node.
			while (it != node.children.end()) {
				if (it->second.x == x) {
					node.children.erase(it);
					return true;
				}
				++it;
			}
			return false;
		}

		// Check the children of this node, and recurse if the children are
		// close enough to contain x.
		while (it != node.children.end() && dist < std::ldexp(1, 1 + it->first)) {
			Node &child = it->second;
			if (x == child.x) {
				Node copy(x); // x is just a convenient value, we won't be keeping this
				copy.children.swap(child.children);
				node.children.erase(it); // child ref now invalid!
				move_children(copy);
				return true;
			}
			if (remove_recursive(x, child))
				return true;
			++it;
		}
		return false;
	}

	template <class OutputIterator>
	OutputIterator recursive_neighbors(OutputIterator out,
		const T &x,
		double margin,
		const Node &node) const
	{
		const double dist = d(x, node.x);
		if (dist < margin)
			*out++ = std::make_pair(dist, node.x);
		typename Node::ChildContainer::const_iterator it = node.children.begin();
		while (it != node.children.end()
			&& dist < margin + std::ldexp(1., 1 + it->first)) {
			out = recursive_neighbors(out, x, margin, it->second);
			++it;
		}
		return out;
	}

	template <class OutputIterator>
	OutputIterator recursive_contents(OutputIterator out, const Node &node) const{
		*out++ = node.x;
		typename Node::ChildContainer::const_iterator it = node.children.begin();
		while (it != node.children.end()) {
			out = recursive_contents(out, it->second);
			++it;
		}
		return out;
	}

	static void print(std::ostream &out, const Node *node) {
		typename Node::ChildContainer::const_iterator it = node->children.begin();
		while (it != node->children.end()) {
			out << it->first << "\t"
				<< node->x << "\t"
				<< it->second.x << "\n";
			print(out, it->second);
			++it;
		}
	}

private:
	Node *root;
	Metric d;
	bool duplicate;
};


typedef struct PyCoverTree {
	PyObject_HEAD
	CoverTree<StateVector, DistanceFunction> * tree;
} PyCoverTreeObject;

extern PyTypeObject PyCoverTree_Type;

#define PyCoverTree_Check(op) PyObject_TypeCheck(op, &PyCoverTree_Type)
#define PyCoverTree_CheckExact(op) (Py_TYPE(op) == &PyCoverTree_Type)

void insert(PyCoverTreeObject * self);
void remove(PyCoverTreeObject * self);

void neighbors(PyCoverTreeObject * self);

#endif	// _COVERTREE_HH_