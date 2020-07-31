// -*- mode: c++; c-basic-offset: 4; -*-
//
// BSpline.cpp: implementation of the BSplineBase class.
//
//////////////////////////////////////////////////////////////////////
/*
 * Copyright (c) 1998,1999
 * University Corporation for Atmospheric Research, UCAR
 *
 * Permission to use, copy, modify, distribute and sell this software and
 * its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and that
 * both that copyright notice and this permission notice appear in
 * supporting documentation.  UCAR makes no representations about the
 * suitability of this software for any purpose.  It is provided "as is"
 * without express or implied warranty.
 *
 * Note from the author:
 *
 * Where possible, you are encouraged to follow the GNU General Public
 * License, or at least the spirit of the license, for the distribution and
 * licensing of this software and any derived works.  See
 * http://www.gnu.org/copyleft/gpl.html.
 */

/**
 * @file
 *
 * This file defines the implementation for the BSpline and BSplineBase
 * templates.
 **/

/*
 * If we're on WIN32 and the DLL macro has not been set yet, which
 * implies the interface file (BSpline.h) has not been included yet,
 * assume only the implementation is wanted and empty the DLL macro
 * before including the interface.
 */
#if WIN32
# ifndef BSPLINE_DLL_
#  define BSPLINE_DLL_
# endif
#endif /* WIN32 */

#include "BSpline.h"
#include "BandedMatrix.h"

#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>
#include <assert.h>

/*
 * This class simulates a namespace for private symbols used by this template
 * implementation which should not pollute the global namespace.
 */
class my
{
public:
    template <class T>
    static inline
    T abs(const T t) { return (t < 0) ? -t : t; }

    template <class T>
    static inline
    const T& min (const T& a, const T& b) { return (a < b) ? a : b; }

    template <class T>
    static inline
    const T& max (const T& a, const T& b) { return (a > b) ? a : b; }
};


template <class T>
class Matrix : public BandedMatrix<T>
{
public:
    Matrix &operator += (const Matrix &B)
    {
	Matrix &A = *this;
        typename Matrix::size_type M = A.num_rows();
	typename Matrix::size_type N = A.num_cols();

	assert(M==B.num_rows());
	assert(N==B.num_cols());

	typename Matrix::size_type i,j;
	for (i=0; i<M; i++)
	    for (j=0; j<N; j++)
		A[i][j] += B[i][j];
	return A;
    }

    inline Matrix & operator= (const Matrix &b)
    {
	return Copy (*this, b);
    }

    inline Matrix & operator= (const T &e)
    {
	BandedMatrix<T>::operator= (e);
	return *this;
    }

};


// Our private state structure, which hides our use of some matrix
// template classes.

template <class T>
struct BSplineBaseP
{
    typedef Matrix<T> MatrixT;

    MatrixT Q;				// Holds P+Q and its factorization
    T* q;
    std::vector<T> X;
    std::vector<T> Nodes;
};


// For now, hardcoding type 1 boundary conditions,
// which constrains the derivative to zero at the endpoints.
template <class T>
const real BSplineBase<T>::BoundaryConditions[9][4] =
{
    //	0	1	M-1	M
    {	-4,	-1,	-1,	-4 },
    {	0,	1,	1,	0 },
    {	2,	-1,	-1,	2 },
    {   -4,     -1,	1,      0 },
    {   -4,     -1,     -1,     2 },
    {   0,      1,      -1,     -4 },
    {   0,      1,      -1,     2 },
    {   2,      -1,     -1,     -4 },
    {   2,      -1,     1,      0 }
};


template <class T>
inline bool BSplineBase<T>::Debug (int on)
{
    static bool debug = false;
    if (on >= 0)
	debug = (on > 0);
    return debug;
}

template <class T>
const real BSplineBase<T>::PI = 3.141592653589793;

template <class T>
const real BSplineBase<T>::ONESIXTH = 1.0 / 6.0;

template <class T>
const char *
BSplineBase<T>::ImplVersion()
{
    return ("$Id: BSpline.cpp 3990 2005-12-02 20:06:13Z granger $");
}


template <class T>
const char *
BSplineBase<T>::IfaceVersion()
{
    return (_BSPLINEBASE_IFACE_ID);
}


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////


template <class T>
BSplineBase<T>::~BSplineBase()
{
    delete[] base->q;
    delete base;
}


// This is a member-wise copy except for replacing our
// private base structure with the source's, rather than just copying
// the pointer.  But we use the compiler's default copy constructor for
// constructing our BSplineBaseP.

template <class T>
BSplineBase<T>::BSplineBase (const BSplineBase<T> &bb) :
    K(bb.K), BC(bb.BC), OK(bb.OK), base(new BSplineBaseP<T>(*bb.base))
{
    xmin = bb.xmin;
    xmax = bb.xmax;
    alpha = bb.alpha;
    waveLength = bb.waveLength;
    DX = bb.DX;
    DXrecip = 1 / DX;
    M = bb.M;
    NX = base->X.size();
}


template <class T>
BSplineBase<T>::BSplineBase (const T *x, int nx, real wl, int bc,
			     int num_nodes) :
    K(3), OK(false), base(new BSplineBaseP<T>)
{
    int qSize = base->Q.num_rows()*base->Q.num_cols();
    base->q = new T[qSize];
    setDomain (x, nx, wl, bc, num_nodes);
}


// Methods


template <class T>
bool
BSplineBase<T>::setDomain (const T *x, int nx, real wl, int bc,
			   int num_nodes)
{
    if (nx <= 0 || x == 0 || wl < 0 || bc < 0 || bc > 8)
    {
	return false;
    }
    OK = false;
    waveLength = wl;
    BC = bc;

    // Copy the x array into our storage.
    base->X.resize (nx);
    std::copy (x, x + nx, base->X.begin());
    NX = base->X.size();
    
    // for(int i = 0; i < NX; i++)
    //	  std::cout << "X[" << i << "] = " << base->X[i] << std::endl;

    // The Setup() method determines the number and size of node intervals.
    if (Setup(num_nodes))
    {
	if (Debug())
	{
	    std::cerr << "Using M node intervals: " << M << " of length DX: "
		      << DX << std::endl;
	    std::cerr << "X min: " << xmin << " ; X max: " << xmax
		      << std::endl;
	    std::cerr << "Data points per interval: " << (float)NX/(float)M
		      << std::endl;
	    std::cerr << "Nodes per wavelength: "
		      << (float)waveLength/(float)DX << std::endl;
	    std::cerr << "Derivative constraint degree: " << K << std::endl;
	}

	// Now we can calculate alpha and our Q matrix
	alpha = Alpha (waveLength);
	if (Debug())
	{
	    std::cerr << "Cutoff wavelength: " << waveLength << " ; "
		      << "Alpha: " << alpha << std::endl;
	    std::cerr << "Calculating Q..." << std::endl;
	}
	calculateQ ();
	if (Debug() && M < 30)
	{
	    std::cerr.fill(' ');
	    std::cerr.precision(2);
	    std::cerr.width(5);
	    std::cerr << base->Q << std::endl;
	}

	if (Debug()) std::cerr << "Calculating P..." << std::endl;
	addP ();
	if (Debug())
	{
	    std::cerr << "Done." << std::endl;
	    if (M < 30)
	    {
		std::cerr << "Array Q after addition of P." << std::endl;
		std::cerr << base->Q;
	    }
	}

	// Now perform the LU factorization on Q
	if (Debug()) std::cerr << "Beginning LU factoring of P+Q..."
			     << std::endl;
	if (! factor ())
	{
	    if (Debug()) std::cerr << "Factoring failed." << std::endl;
	}
	else
	{
	    if (Debug()) std::cerr << "Done." << std::endl;
	    OK = true;
		int qSize = base->Q.num_rows()*base->Q.num_cols();
		delete[] base->q;
		base->q = new T[qSize];
	}
    }
    return OK;
}

template <class T>
bool
BSplineBase<T>::setDomainGQ (const T *x, int nx, real wl, int bc,
						   int num_nodes)
{
    if (nx <= 0 || x == 0 || wl < 0 || bc < 0 || bc > 8)
    {
		return false;
    }
    OK = false;
    waveLength = wl;
    BC = bc;

    // Copy the x array into our storage.
    base->X.resize (nx);
    std::copy (x, x+nx, base->X.begin());
    NX = base->X.size();

    // The Setup() method determines the number and size of node intervals.
    if (SetupGQ(num_nodes))
    {
		if (Debug())
		{
			std::cerr << "Using M node intervals: " << M << " of length DX: "
			<< DX << std::endl;
			std::cerr << "X min: " << xmin << " ; X max: " << xmax
			<< std::endl;
			std::cerr << "Data points per interval: " << (float)NX/(float)M
			<< std::endl;
			std::cerr << "Nodes per wavelength: "
			<< (float)waveLength/(float)DX << std::endl;
			std::cerr << "Derivative constraint degree: " << K << std::endl;
		}

		// Now we can calculate alpha and our Q matrix
		alpha = AlphaGQ (waveLength);
		if (Debug())
		{
			std::cerr << "Cutoff wavelength: " << waveLength << " ; "
			<< "Alpha: " << alpha << std::endl;
			std::cerr << "Calculating Q..." << std::endl;
		}
		calculateQ ();
		if (Debug() && M < 30)
		{
			std::cerr.fill(' ');
			std::cerr.precision(2);
			std::cerr.width(5);
			std::cerr << base->Q << std::endl;
		}

		if (Debug()) std::cerr << "Calculating P..." << std::endl;
		addPGQ ();
		if (Debug())
		{
			std::cerr << "Done." << std::endl;
			if (M < 100)
			{
				std::cerr << "Array Q after addition of P." << std::endl;
				std::cerr << base->Q;
			}
		}

		// Now perform the LU factorization on Q
		if (Debug()) std::cerr << "Beginning LU factoring of P+Q..."
			<< std::endl;
		if (! factor ())
		{
			if (Debug()) std::cerr << "Factoring failed." << std::endl;
		}
		else
		{
			if (Debug()) std::cerr << "Done." << std::endl;
			OK = true;
			int qSize = base->Q.num_rows()*base->Q.num_cols();
			delete[] base->q;
			base->q = new T[qSize];
		}

    }
    return OK;
}


/*
 * Calculate the alpha parameter given a wavelength.
 */
template <class T>
real
BSplineBase<T>::Alpha (real wl)
{
    // K is the degree of the derivative constraint: 1, 2, or 3
    real a = (real) (wl / (2 * PI * DX));
    a *= a;			// a^2
    if (K == 2)
	a = a * a;		// a^4
    else if (K == 3)
	a = a * a * a;		// a^6
    return a;
}

template <class T>
real
BSplineBase<T>::AlphaGQ (real wl)
{
    // K is the degree of the derivative constraint: 1, 2, or 3
    real a = (real) (wl / (2 * PI * DX));
    a *= a;			// a^2
    if (K == 2)
		a = a * a;		// a^4
    else if (K == 3)
		a = a * a * a;		// a^6
    return a * 0.5;
}


/*
 * Return the correct beta value given the node index.  The value depends
 * on the node index and the current boundary condition type.
 */

template <class T>
inline real
BSplineBase<T>::Beta (/* unsigned */ int m) // TODO called with int
{
    if (m > 1 && m < M-1)
	return 0.0;
    if (m >= M-1)
	m -= M-3;
    assert (0 <= BC && BC <= 8);
    assert (0 <= m && m <= 3);
    return BoundaryConditions[BC][m];
}

/*
 * Given an array of y data points defined over the domain
 * of x data points in this BSplineBase, create a BSpline
 * object which contains the smoothed curve for the y array.
 */
template <class T>
BSpline<T> *
BSplineBase<T>::apply (const T *y)
{
    return new BSpline<T> (*this, y);
}


/*
 * Evaluate the closed basis function at node m for value x,
 * using the parameters for the current boundary conditions.
 */
template <class T>
real
BSplineBase<T>::Basis (int m, T x)
{
    real y = 0;
    real xm = xmin + (m * DX);
	real delta = (real)(x - xm) * (real)DXrecip;
    real z = my::abs(delta);
	//real z = abs(delta);
    if (z < 2.0)
    {
	z = 2 - z;
	y = (z*z*z) * ONESIXTH;
	z -= 1.0;
	if (z > 0)
	    y -= (z*z*z) * 4 * ONESIXTH;
    }

    // Boundary conditions, if any, are an additional addend.
    if (m == 0 || m == 1)
	y += Beta(m) * Basis (-1, x);
    else if (m == M-1 || m == M)
	y += Beta(m) * Basis (M+1, x);

    return y;
}



/*
 * Evaluate the deriviative of the closed basis function at node m for
 * value x, using the parameters for the current boundary conditions.
 */
template <class T>
real
BSplineBase<T>::DBasis (int m, T x)
{
    real dy = 0;
    real xm = xmin + (m * DX);
    real delta = (real)(x - xm) * (real)DXrecip;
    real z = my::abs(delta);
    if (z < 2.0)
    {
	z = 2.0 - z;
	dy = z * z * ONESIXTH;
	z -= 1.0;

	if (z > 0)
	{
	    dy -= z * z * 4 * ONESIXTH;
	}
	dy *= ((delta > 0) ? -1.0 : 1.0) * 3.0 * (real)DXrecip;
    }

    // Boundary conditions, if any, are an additional addend.
    if (m == 0 || m == 1)
	dy += Beta(m) * DBasis (-1, x);
    else if (m == M-1 || m == M)
	dy += Beta(m) * DBasis (M+1, x);

    return dy;
}




template <class T>
real
BSplineBase<T>::qDelta (int m1, int m2)
/*
 * Return the integral of the product of the basis function derivative
 * restricted to the node domain, 0 to M.
 */
{
    // These are the products of the Kth derivative of the
    // normalized basis functions
    // given a distance m nodes apart, qparts[K-1][m], 0 <= m <= 3
    // Each column is the integral over each unit domain, -2 to 2
    static const real qparts[3][4][4] =
    {
	{
	    { 0.11250,   0.63750,   0.63750,   0.11250 },
	    { 0.00000,   0.13125,  -0.54375,   0.13125 },
	    { 0.00000,   0.00000,  -0.22500,  -0.22500 },
	    { 0.00000,   0.00000,   0.00000,  -0.01875 }
	},
	{
	    { 0.75000,   2.25000,   2.25000,   0.75000 },
	    { 0.00000,  -1.12500,  -1.12500,  -1.12500 },
	    { 0.00000,   0.00000,   0.00000,   0.00000 },
	    { 0.00000,   0.00000,   0.00000,   0.37500 }
	},
	{
	/*  { 2.25000,  20.25000,  20.25000,   2.25000 },
	    { 0.00000,  -6.75000, -20.25000,  -6.75000 },
	    { 0.00000,   0.00000,   6.75000,   6.75000 },
	    { 0.00000,   0.00000,   0.00000,  -2.25000 } */

		{ 1.00000,   9.00000,   9.00000,   1.00000 },
	    { 0.00000,  -3.00000,  -9.00000,  -3.00000 },
	    { 0.00000,   0.00000,   3.00000,   3.00000 },
	    { 0.00000,   0.00000,   0.00000,  -1.00000 }
	}
    };

    if (m1 > m2)
	std::swap (m1, m2);

    if (m2 - m1 > 3)
	return 0.0;

    real q = 0;
    for (int m = my::max (m1-2,0); m < my::min (m1+2, M); ++m)
	q += qparts[K-1][m2-m1][m-m1+2];
    return q * alpha;
}

template <class T>
void
BSplineBase<T>::calculateQ ()
{
    Matrix<T> &Q = base->Q;
    Q.setup (M+1, 3);
    Q = 0;
    if (alpha == 0)
	return;

    // First fill in the q values without the boundary constraints.
    int i;
    for (i = 0; i <= M; ++i)
    {
	Q[i][i] = qDelta(i,i);
	for (int j = 1; j < 4 && i+j <= M; ++j)
	{
	    Q[i][i+j] = Q[i+j][i] = qDelta (i, i+j);
	}
    }

    // Now add the boundary constraints:
    // First the upper left corner.
    float b1, b2, q;
    for (i = 0; i <= 1; ++i)
    {
	b1 = Beta(i);
	for (int j = i; j < i+4; ++j)
	{
	    b2 = Beta(j);
	    assert (j-i >= 0 && j - i < 4);
	    q = 0.0;
	    if (i+1 < 4)
		q += b2*qDelta(-1,i);
	    if (j+1 < 4)
		q += b1*qDelta(-1,j);
	    q += b1*b2*qDelta(-1,-1);
	    Q[j][i] = (Q[i][j] += q);
	}
    }

    // Then the lower right
    for (i = M-1; i <= M; ++i)
    {
	b1 = Beta(i);
	for (int j = i - 3; j <= i; ++j)
	{
	    b2 = Beta(j);
	    q = 0.0;
	    if (M+1-i < 4)
		q += b2*qDelta(i,M+1);
	    if (M+1-j < 4)
		q += b1*qDelta(j,M+1);
	    q += b1*b2*qDelta(M+1,M+1);
	    Q[j][i] = (Q[i][j] += q);
	}
    }
}

template <class T>
void
BSplineBase<T>::addP ()
{
    // Add directly to Q's elements
    Matrix<T> &P = base->Q;
    std::vector<T> &X = base->X;

    // For each data point, sum the product of the nearest, non-zero Basis
    // nodes
    int mx, m, n, i;
    for (i = 0; i < NX; ++i)
    {
	// Which node does this put us in?
	T &x = X[i];
	mx = (int)((x - xmin) / DX);

	// Loop over the upper triangle of nonzero basis functions,
	// and add in the products on each side of the diagonal.
	for (m = my::max(0, mx-1); m <= my::min(M, mx+2); ++m)
	{
	    float pn;
	    float pm = Basis (m, x);
	    float sum = pm * pm;
	    P[m][m] += sum;
	    for (n = m+1; n <= my::min(M, mx+2); ++n)
	    {
		pn = Basis (n, x);
		sum = pm * pn;
		P[m][n] += sum;
		P[n][m] += sum;
	    }
	}
    }
}

template <class T>
void
BSplineBase<T>::addPGQ ()
{
    // Add directly to Q's elements
    Matrix<T> &P = base->Q;
    std::vector<T> &X = base->X;

    // For each data point, sum the product of the nearest, non-zero Basis
    // nodes
    int mx, m, n, i;
    for (i = 0; i < NX; ++i)
    {
		// Which node does this put us in?
		T &x = X[i];
		mx = (int)((x - xmin) / DX);

		// Loop over the upper triangle of nonzero basis functions,
		// and add in the products on each side of the diagonal.
		for (m = my::max(0, mx-1); m <= my::min(M, mx+2); ++m)
		{
			float pn;
			float pm = Basis (m, x);
			float sum = pm * pm * 0.5;
			P[m][m] += sum;
			for (n = m+1; n <= my::min(M, mx+2); ++n)
			{
				pn = Basis (n, x);
				sum = pm * pn * 0.5;
				P[m][n] += sum;
				P[n][m] += sum;
			}
		}
    }
}


template <class T>
bool
BSplineBase<T>::factor ()
{
    Matrix<T> &LU = base->Q;

    if (LU_factor_banded (LU, 3) != 0)
    {
        if (Debug()) std::cerr << "LU_factor_banded() failed." << std::endl;
	return false;
    }
    if (Debug() && M < 30)
	std::cerr << "LU decomposition: " << std::endl << LU << std::endl;
    return true;
}



template <class T>
inline real
BSplineBase<T>::Ratiod (int &ni, real &deltax, real &ratiof)
{
    deltax = (xmax - xmin) / ni;
    ratiof = waveLength / deltax;
    real ratiod = (real) NX / (real) (ni + 1);
    return ratiod;
}


// Setup the number of nodes (and hence deltax) for the given domain and
// cutoff wavelength.  According to Ooyama, the derivative constraint
// approximates a lo-pass filter if the cutoff wavelength is about 4*deltax
// or more, but it should at least be 2*deltax.  We can increase the number
// of nodes to increase the number of nodes per cutoff wavelength.
// However, to get a reasonable representation of the data, the setup
// enforces at least as many nodes as data points in the domain.  (This
// constraint assumes reasonably even distribution of data points, since
// its really the density of data points which matters.)
//
// Return zero if the setup fails, non-zero otherwise.
//
// The algorithm in this routine is mostly taken from the FORTRAN
// implementation by James Franklin, NOAA/HRD.
//
template <class T>
bool
BSplineBase<T>::Setup(int num_nodes)
{
    std::vector<T> &X = base->X;

    // Find the min and max of the x domain
    xmin = X[0];
    xmax = X[0];

    // std::cout << "::Setup: X[0] = " << X[0] << std::endl;
    
    int i;
    for (i = 1; i < NX; ++i)
    {
	if (X[i] < xmin)
	    xmin = X[i];
	else if (X[i] > xmax)
	    xmax = X[i];
    }

    int ni = 9;			// Number of node intervals (NX - 1)
    real deltax;

    if (num_nodes >= 2)
    {
	// We've been told explicitly the number of nodes to use.
	ni = num_nodes - 1;
	if (waveLength == 0)
	{
	    waveLength = 1.0;
	}
    }
    else if (waveLength == 0)
    {
	// Turn off frequency constraint and just set two node intervals per
	// data point.
	ni = NX * 2;
	waveLength = 1;
    }
    else if (waveLength > xmax - xmin)
    {
	return (false);
    }
    else
    {
	// Minimum acceptable number of node intervals per cutoff wavelength.
	static const real fmin = 2.0;

	real ratiof;	// Nodes per wavelength for current deltax
	real ratiod;	// Points per node interval

	// Increase the number of node intervals until we reach the minimum
	// number of intervals per cutoff wavelength, but only as long as
	// we can maintain at least one point per interval.
	do {
	    if (Ratiod (++ni, deltax, ratiof) < 1.0)
		return false;
	}
	while (ratiof < fmin);

	// Now increase the number of intervals until we have at least 4
	// intervals per cutoff wavelength, but only as long as we can
	// maintain at least 2 points per node interval.  There's also no
	// point to increasing the number of intervals if we already have
	// 15 or more nodes per cutoff wavelength.
	//
	do {
	    if ((ratiod = Ratiod (++ni, deltax, ratiof)) < 1.0 ||
		ratiof > 15.0)
	    {
		--ni;
		break;
	    }
	}
	while (ratiof < 4 || ratiod > 2.0);
    }

    // Store the calculations in our state
    M = ni;
    DX = (xmax - xmin) / ni;
    DXrecip = 1 / DX;
    return (true);
}

template <class T>
bool
BSplineBase<T>::SetupGQ(int num_nodes)
{
    std::vector<T> &X = base->X;

    // Find the min and max of the x domain
	float mu = 0.5 - 0.5*sqrt(1. / 3.);
    xmin = X[0] - mu;
    xmax = X[0]  +mu;

    int i;
    for (i = 1; i < NX; ++i)
    {
		if ((X[i]  -mu) < xmin)
			xmin = X[i] - mu;
		else if ((X[i]+mu) > xmax)
			xmax = X[i] + mu;
    }

    int ni = 9;			// Number of node intervals (NX - 1)
    real deltax;

    if (num_nodes >= 2)
    {
		// We've been told explicitly the number of nodes to use.
		ni = num_nodes - 1;
		if (waveLength == 0)
		{
			waveLength = 1.0;
		}
    }
    else if (waveLength == 0)
    {
		// Turn off frequency constraint and just set two node intervals per
		// data point.
		ni = NX * 2;
		waveLength = 1;
    }
    else if (waveLength > xmax - xmin)
    {
		return (false);
    }
    else
    {
		// Minimum acceptable number of node intervals per cutoff wavelength.
		static const real fmin = 2.0;

		real ratiof;	// Nodes per wavelength for current deltax
		real ratiod;	// Points per node interval

		// Increase the number of node intervals until we reach the minimum
		// number of intervals per cutoff wavelength, but only as long as
		// we can maintain at least one point per interval.
		do {
			if (Ratiod (++ni, deltax, ratiof) < 1.0)
				return false;
		}
		while (ratiof < fmin);

		// Now increase the number of intervals until we have at least 4
		// intervals per cutoff wavelength, but only as long as we can
		// maintain at least 2 points per node interval.  There's also no
		// point to increasing the number of intervals if we already have
		// 15 or more nodes per cutoff wavelength.
		//
		do {
			if ((ratiod = Ratiod (++ni, deltax, ratiof)) < 1.0 ||
				ratiof > 15.0)
			{
				--ni;
				break;
			}
		}
		while (ratiof < 4 || ratiod > 2.0);
    }

    // Store the calculations in our state
    M = ni;
    DX = (xmax - xmin) / ni;
	DXrecip = 1 / DX;

    return (true);
}


template <class T>
const T *
BSplineBase<T>::nodes (int *nn)
{
    if (base->Nodes.size() == 0)
    {
	base->Nodes.reserve (M+1);
	for (int i = 0; i <= M; ++i)
	{
	    base->Nodes.push_back ( xmin + (i * DX) );
	}
    }

    if (nn)
	*nn = base->Nodes.size();

    assert (base->Nodes.size() == (unsigned)(M+1));
    return &base->Nodes[0];
}



template <class T>
std::ostream &operator<< (std::ostream &out, const std::vector<T> &c)
{
    for (typename std::vector<T>::const_iterator it = c.begin(); it < c.end(); ++it)
	out << *it << ", ";
    out << std::endl;
    return out;
}



//////////////////////////////////////////////////////////////////////
// BSpline Class
//////////////////////////////////////////////////////////////////////

template <class T>
struct BSplineP
{
    std::vector<T> spline;
    std::vector<T> A;
};


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////


/*
 * This BSpline constructor constructs and sets up a new base and
 * solves for the spline curve coeffiecients all at once.
 */
template <class T>
BSpline<T>::BSpline (const T *x, int nx, const T *y,
		     real wl, int bc_type, int num_nodes) :
    BSplineBase<T>(x, nx, wl, bc_type, num_nodes), s(new BSplineP<T>)
{
    solve (y);
}



/*
 * Create a new spline given a BSplineBase.
 */
template <class T>
BSpline<T>::BSpline (BSplineBase<T> &bb, const T *y) :
    BSplineBase<T>(bb), s(new BSplineP<T>)
{
    solve (y);
}

/*
 * Default constructor for array declaration
 */
template <class T>
BSpline<T>::BSpline () :
BSplineBase<T>(0, 0, 0., 0, 0), s(new BSplineP<T>)
{
}


/*
 * (Re)calculate the spline for the given set of y values.
 */
template <class T>
bool
BSpline<T>::solve (const T *y)
{
    //if (! OK)
	//return false;

    // Any previously calculated curve is now invalid.
    s->spline.clear ();
    OK = false;

    // Given an array of data points over x and its precalculated
    // P+Q matrix, calculate the b vector and solve for the coefficients.
    std::vector<T> &B = s->A;
    std::vector<T> &A = s->A;
    A.clear ();
    A.resize (M+1);

    if (Debug()) std::cerr << "Solving for B..." << std::endl;

    // Find the mean of these data
    mean = 0.0;
    /* int i;
    for (i = 0; i < NX; ++i)
    {
	mean += y[i];
    }
    mean = mean / (real)NX; */
    if (Debug())
	std::cerr << "Mean for y: " << mean << std::endl;

    int mx, m, j;
    for (j = 0; j < NX; ++j)
    {
	// Which node does this put us in?
	T &xj = base->X[j];
	T yj = y[j] - mean;
	mx = (int)((xj - xmin) / DX);

	for (m = my::max(0,mx-1); m <= my::min(mx+2,M); ++m)
	{
	    B[m] += yj * this->Basis (m, xj);
	}
    }

    if (Debug() && M < 30)
    {
	std::cerr << "Solution a for (P+Q)a = b" << std::endl;
	std::cerr << " b: " << B << std::endl;
    }

    // Now solve for the A vector in place.
    if (LU_solve_banded (base->Q, A, 3) != 0)
    {
	if (Debug())
	    std::cerr << "LU_solve_banded() failed." << std::endl;
    }
    else
    {
	OK = true;
	if (Debug()) std::cerr << "Done." << std::endl;
	if (Debug() && M < 30)
	{
	    std::cerr << " a: " << A << std::endl;
	    std::cerr << "LU factor of (P+Q) = " << std::endl
		      << base->Q << std::endl;
	}
    }
    return (OK);
}

/*
 * (Re)calculate the spline for the given set of y values using 2-pt Gaussian quadrature
 */
template <class T>
bool
BSpline<T>::solveGQ (const T *y)
{
    //if (! OK)
	//	return false;

    // Any previously calculated curve is now invalid.
    s->spline.clear ();
    OK = false;

    // Given an array of data points over x and its precalculated
    // P+Q matrix, calculate the b vector and solve for the coefficients.
    std::vector<T> &B = s->A;
    std::vector<T> &A = s->A;
    A.clear ();
    A.resize (M+1);

    if (Debug()) std::cerr << "Solving for B..." << std::endl;

    // Find the mean of these data
    mean = 0.0;
    /* int i;
	 for (i = 0; i < NX; ++i)
	 {
	 mean += y[i];
	 }
	 mean = mean / (real)NX; */
    if (Debug())
		std::cerr << "Mean for y: " << mean << std::endl;

    int mx, m, j;
    for (j = 0; j < NX; ++j)
    {
		// Which node does this put us in?
		T &xj = base->X[j];
		T yj = y[j] - mean;
		mx = (int)((xj - xmin) / DX);
		// Assuming that the xj array is set up on GQ points
		for (m = my::max(0,mx-1); m <= my::min(mx+2,M); ++m)
		{
			B[m] += yj * this->Basis (m, xj) * 0.5;
		}
    }

    if (Debug() && M < 100)
    {
		std::cerr << "Solution a for (P+Q)a = b" << std::endl;
		std::cerr << " b: " << B << std::endl;
    }

    // Now solve for the A vector in place.
    if (LU_solve_banded (base->Q, A, 3) != 0)
    {
		if (Debug())
			std::cerr << "LU_solve_banded() failed." << std::endl;
    }
    else
    {
		OK = true;
		if (Debug()) std::cerr << "Done." << std::endl;
		if (Debug() && M < 100)
		{
			std::cerr << " a: " << A << std::endl;
			std::cerr << "LU factor of (P+Q) = " << std::endl
			<< base->Q << std::endl;
		}
    }
    return (OK);
}

/*
 * (Re)calculate the spline for the given set of y values using 2-pt Gaussian quadrature
 */
template <class T>
bool
BSpline<T>::solveBGQ (const T *b)
{
    //if (! OK)
	//	return false;

    // Any previously calculated curve is now invalid.
    s->spline.clear ();
    OK = false;

    // Given an array of data points over x and its precalculated
    // P+Q matrix, calculate the b vector and solve for the coefficients.
    std::vector<T> &B = s->A;
    std::vector<T> &A = s->A;
    A.clear ();
    A.resize (M+1);

    if (Debug()) std::cerr << "Solving for B..." << std::endl;

    // Find the mean of these data
    mean = 0.0;
    /* int i;
	 for (i = 0; i < NX; ++i)
	 {
	 mean += y[i];
	 }
	 mean = mean / (real)NX; */
    if (Debug())
		std::cerr << "Mean for y: " << mean << std::endl;

    int m;
	for (m = 0; m < M+1; ++m) {
		B[m] = b[m];
	}

    if (Debug() && M < 100)
    {
		std::cerr << "Solution a for (P+Q)a = b" << std::endl;
		std::cerr << " b: " << B << std::endl;
    }

    // Now solve for the A vector in place.
    if (LU_solve_banded (base->Q, A, 3) != 0)
    {
		if (Debug())
			std::cerr << "LU_solve_banded() failed." << std::endl;
    }
    else
    {
		OK = true;
		if (Debug()) std::cerr << "Done." << std::endl;
		if (Debug() && M < 100)
		{
			std::cerr << " a: " << A << std::endl;
			std::cerr << "LU factor of (P+Q) = " << std::endl
			<< base->Q << std::endl;
		}
    }
    return (OK);
}

/*
 * Solve for the inverse spline problem given a set of coefficients, get back a set of x values
 */
template <class T>
const T*
BSpline<T>::solveInverseGQ (const T *b)
{
    if (! OK)
		return NULL;

    // Any previously calculated curve is now invalid.
    s->spline.clear ();
    OK = false;

    // Given an array of data points over x and its precalculated
    // P+Q matrix, calculate the b vector and solve for the coefficients.
    std::vector<T> &B = s->A;
    std::vector<T> &A = s->A;
    A.clear ();
    A.resize (M+1);

    if (Debug()) std::cerr << "Solving for B..." << std::endl;

    // Find the mean of these data
    mean = 0.0;
    /* int i;
	 for (i = 0; i < NX; ++i)
	 {
	 mean += y[i];
	 }
	 mean = mean / (real)NX; */
    if (Debug())
		std::cerr << "Mean for y: " << mean << std::endl;

    int mx, m, j;
	// y should have same dimension as nodes
	for (m = 0; m < M+1; ++m) {
		B[m] = b[m];
	}

    if (Debug() && M < 60)
    {
		std::cerr << "Solution a for (P+Q)a = b" << std::endl;
		std::cerr << " b: " << B << std::endl;
    }

    // Now solve for the A vector in place.
    if (LU_solve_banded (base->Q, A, 3) != 0)
    {
		if (Debug())
			std::cerr << "LU_solve_banded() failed." << std::endl;
    }
    else
    {
		OK = true;
		if (Debug()) std::cerr << "Done." << std::endl;
		if (Debug() && M < 60)
		{
			std::cerr << " a: " << A << std::endl;
			/* std::cerr << "LU factor of (P+Q) = " << std::endl
			<< base->Q << std::endl; */
		}
    }

	// Now invert the SB transform
	std::vector<T> &spline = s->spline;
	for (j = 0; j < NX; ++j)
    {
		T y = 0;
		// Which node does this put us in?
		T &xj = base->X[j];
		mx = (int)((xj - xmin) / DX);
		for (m = my::max(0,mx-1); m <= my::min(mx+2,M); ++m)
		{
			y += B[m] * this->Basis (m, xj) * 0.5;
		}
		// Put the results in the hidden spline vector
		s->spline.push_back(y);
    }

	return &spline[0];
}

template <class T>
BSpline<T>::~BSpline()
{
    delete s;
}

template <class T>
const T* BSpline<T>::getQfactored()
{

	if(dumpMatrixToArray(base->Q, base->q)) return base->q;
	return 0;
}

template <class T>
T BSpline<T>::getCoefficient (int n)
{
    if (OK)
	if (0 <= n && n <= M)
	    return s->A[n];
    return 0;
}

template <class T>
bool BSpline<T>::setCoefficient (int n, T coeff)
{
    if (OK)
		if (0 <= n && n <= M) {
			s->A[n] = coeff;
			return true;
		}
    return 0;
}

template <class T>
const T *BSpline<T>::getBGQ (const T *y)
{
	//if (! OK)
	//	return false;

    // Any previously calculated curve is now invalid.
    s->spline.clear ();
    OK = false;

    // Given an array of data points over x and its precalculated
    // P+Q matrix, calculate the b vector and solve for the coefficients.
    std::vector<T> &B = s->A;
    std::vector<T> &A = s->A;
    A.clear ();
    A.resize (M+1);

    if (Debug()) std::cerr << "Solving for B..." << std::endl;

    // Find the mean of these data
    mean = 0.0;
    /* int i;
	 for (i = 0; i < NX; ++i)
	 {
	 mean += y[i];
	 }
	 mean = mean / (real)NX; */
    if (Debug())
		std::cerr << "Mean for y: " << mean << std::endl;

    int mx, m, j;
    for (j = 0; j < NX; ++j)
    {
		// Which node does this put us in?
		T &xj = base->X[j];
		T yj = y[j] - mean;
		mx = (int)((xj - xmin) / DX);
		// Assuming that the xj array is set up on GQ points
		for (m = my::max(0,mx-1); m <= my::min(mx+2,M); ++m)
		{
			B[m] += yj * this->Basis (m, xj) * 0.5;
		}
    }

	return &B[0];
}

template <class T>
T BSpline<T>::evaluate (T x)
{
    T y = 0;
    if (OK)
    {
	int n = (int)((x - xmin)/DX);
	for (int i = my::max(0,n-1); i <= my::min(M,n+2); ++i)
	{
	    y += s->A[i] * this->Basis (i, x);
	}
	y += mean;
    }
    return y;
}



template <class T>
T BSpline<T>::slope (T x)
{
    T dy = 0;
    if (OK)
    {
	int n = (int)((x - xmin)/DX);

	for (int i = my::max(0,n-1); i <= my::min(M,n+2); ++i)
	{
	    dy += s->A[i] * this->DBasis (i, x);
	}
    }
    return dy;
}

template <class T>
T BSpline<T>::getBasis (int n, T x)
{
	return this->Basis (n, x);
}

template <class T>
T BSpline<T>::getDBasis (int n, T x)
{
	return this->DBasis (n, x);
}


template <class T>
const T *BSpline<T>::curve (int *nx)
{
    if (! OK)
	return 0;

    // If we already have the curve calculated, don't do it again.
    std::vector<T> &spline = s->spline;
    if (spline.size() == 0)
    {
	spline.reserve (M+1);
	for (int n = 0; n <= M; ++n)
	{
	    T x = xmin + (n * DX);
	    spline.push_back (evaluate (x));
	}
    }

    if (nx)
	*nx = spline.size();
    return &spline[0];
}
