/*
 *  CostFunction.h
 *  SAMURAI
 *
 *  Copyright 2010 Michael Bell. All rights reserved.
 *
 */

#ifndef COSTFUNC_H
#define COSTFUNC_H
#include "precision.h"
#include "Projection.h"

#ifdef PETSC
#include "petsctao.h"
#include "petscksp.h"
#endif

using namespace std;


class CostFunction; // forward declaration so we can use it in AppCtx

//for Petsc solver
#ifdef PETSC
typedef struct {
  PetscScalar  *q;
  PetscScalar  *xi;
  Vec          diag;
  int          *ind;
  Mat          H;
  CostFunction *cf;
} AppCtx;
#endif

class CostFunction
{

public:
	
  CostFunction(const Projection& proj, const int& numObs = 0, const int& stateSize = 0);
	virtual ~CostFunction();
	void setNumObservations(const int& numObs);
	int getNumObservations();
	void setLengthStateVector(const int& stateSize);
	int getLengthStateVector();
	bool minimize();
	

	int ls_cnt; 
	int mObs;
	int nState;
	real* currState;
	real* currGradient;
	real* tempState;
	real* tempGradient;
	real* xt;
	real* df;
	real* mt_work;
	const Projection& projection;

	//new PETSC solver stuff	       
	virtual real funcValue(real* state) = 0;
	virtual void funcGradient(real* state, real* gradient) = 0;
	virtual real funcValueAndGradient(real* state, real* gradient) = 0;
	virtual void funcHessian(real *x, real *hessian) = 0;
#ifdef PETSC
	void callPetscSolver(real *q, real* xi, const real ftol);
#endif
	void truncatedNewton(real* q, real* xi, const real ftol);
	void conjugateGradient(real* q, real* xi, const real ftol, real funcMin);
	void dlinmin(real* &p, real* &xi, real &fretp);
	real f1dim(const real x);
	real df1dim(const real x);
	real f1dim_and_df1dim(const real x, real *grad);
	inline void mov3(real &a, real &b, real &c,
					 const real d, const real e, const real f);
	real dbrent(const real ax, const real bx, const real cx,
				  const real tol, real &xmin);
	inline void shft3(real &a, real &b, real &c, const real d);
	void mnbrack(real &ax, real &bx, real &cx, 
				 real &fa, real &fb, real &fc);
	int MTLineSearch(real* &x, real* &g, real *s, real *fval, real initstep);
	int MTcstep(real *stx, real *fx, real *dx, real* sty, real *fy, real *dy, 
		    real *stp, real *fp, real *dp, int *bracket, real *stepmin, 
		    real *stepmax);

#pragma acc declare copyin(mObs,nState)	
private:
	inline real CF_MAX(const real &a, const real &b) 
	{return b > a ? (b) : (a); }
	inline real CF_MIN(const real &a, const real &b) 
	{return b < a ? (b) : (a); }
	inline real CF_SIGN(const real &a, const real &b) 
	{return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a); }
	inline void CF_SWAP(real &a, real &b) 
	{real dum=a; a=b; b=dum;}
	
};

#endif

