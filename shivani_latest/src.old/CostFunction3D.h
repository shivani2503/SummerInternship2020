/*
 *  CostFunction3D.h
 *  samurai
 *
 *  Copyright 2008 Michael Bell. All rights reserved.
 *
 */

#ifndef COSTFUNC3D_H
#define COSTFUNC3D_H

#include "CostFunction.h"
#include "BSpline.h"
#include "RecursiveFilter.h"
#include "Observation.h"
#include "ReferenceState.h"
// #include "VarDriver3D.h"
#include "ErrorData.h"

#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <string>
#include <unordered_map>
//#include <complex.h>
#include <fftw3.h>

class CostFunction3D : public CostFunction
{

public:

CostFunction3D(const Projection& proj, const int& numObs = 0, const int& stateSize = 0);
	virtual ~CostFunction3D();
    void initialize(std::unordered_map<std::string, std::string>* config, real* bgU, real* obs, ReferenceState* ref);
	void finalize();
	void updateBG();
	void initState(const int iteration);
	bool copyResults(int iDim, int jDim, int kDim,
			 float *u, float *v, float *w, float *th, float *p);

//protected:
	double funcValue(double* state);
	void funcGradient(double* state, double* gradient);
	//extra for Petsc solver
	double funcValueAndGradient(double *state, double *gradient);
	void funcHessian(double *x, double *hessian);
	// end extra for Petsc solver
	void updateHCq(double* state, double* HCq);
	real Basis(const int& m, const real& x, const int& M,const real& xmin,
			   const real& DX, const real& DXrecip, const int& derivative,
			   const int& BL, const int& BR, const real& lambda = 0);
	real BasisBC(real b, const int& m, const real& x, const int& M,const real& xmin,
			   const real& DX, const real& DXrecip, const int& derivative,
			   const int& BL, const int& BR, const real& lambda = 0);
	void fillBasisLookup();
	bool filterArray(real* array, const int& arrLength);
	bool setupSplines();
	void obAdjustments();
	void solveBC(real* A, real* B);
	bool SAtransform(const real* Bstate, real* Astate);
	bool SAtranspose(const real* Astate, real* Bstate);
	void calcInnovation();
	void calcHTranspose(const real* yhat, real* Astate);
        void calcHTransposeOld(const real* yhat, real* Astate);
        bool outputAnalysis(const std::string& suffix, real* Astate) { return false;}
       //virtual bool outputAnalysis(const std::string& suffix, real* Astate) = 0;
	void SBtransform(const real* Ustate, real* Bstate);
	void SBtranspose(const real* Bstate, real* Ustate);
        void SCtransform(const real* Astate, real* Cstate);
	void SCtranspose(const real* Cstate, real* Astate);
	void FFtransform(const real* Astate, real* Cstate);

	bool writeAsi(const std::string& asiFileName);
	bool writeNetCDF(const std::string& netcdfFileName);
	
	void adjustInternalDomain(int increment);
	void calcSplineCoefficients(const int& Dim, const real& eq, const int* BCL, const int* BCR,
                                const real& xmin, const real& DX, const real& DXrecip, const int& LDim,
                                real* L[7], real* gamma[7]);
	bool copy3DArray(real *src, float *dest, int iDim, int jDim, int kDim);
	void calcHmatrix();
	void Htransform(const real* Cstate, real* Hstate);

	// A couple of utilities functions to help query config values
  /*fixme bool isTrue(const char *flag_in) const {
    std::string flag = flag_in;
    if (configHash->find(flag) != configHash->end()) {
      if ((*configHash)[flag] == "true") {
        return true;
      }
    }
    return false;
  }
	bool isEqual(const char *flag, std::string value) {
	  return configHash->contains(flag) && configHash->value(flag) == value;
	}*/
	
	void initBkgdErrors();
	
	bool mishFlag;
	int iDim, jDim, kDim;
	int iLDim, jLDim, kLDim;
	int iRank[7], jRank[7], kRank[7];
        int kRankMax;
	real iMin, iMax, DI, DIrecip;
	real jMin, jMax, DJ, DJrecip;
	real kMin, kMax, DK, DKrecip;
	real* bgFields;
	real* bgState;
	real* bgStdDev;
	real* obsVector;
        real* obsData;  // This only contains data that is needed by the calcHTranspose2 subroutine
	real* rawObs;
	real* stateA;
	real* stateB;
        //extern "C" {
        real* stateC;
        //}

	real* stateU;
        real* tempHval;
	real* CTHTd;
	real* HCq;
	real* innovation;
	real* iL[7];
	real* jL[7];
	real* kL[7];
	real* iGamma[7];
	real* jGamma[7];
	real* kGamma[7];
        real * kGammaL, * kLL;
	real* finalAnalysis;
	int varDim;
	int derivDim;
	real bgError[7];
	int iBCL[7], iBCR[7], jBCL[7], jBCR[7], kBCL[7], kBCR[7];
	int derivative[4][3];
	real constHeight;
	real mcWeight;
	int iMaxWavenumber[7], jMaxWavenumber[7], kMaxWavenumber[7];
	double *iFFTin, *jFFTin, *kFFTin;
	fftw_complex *iFFTout, *jFFTout, *kFFTout;
	fftw_plan iForward, jForward, iBackward, jBackward, kForward, kBackward;
	real *H;
	integer *IH, *I2H,*JH;
        integer *mPtr, *mVal;
		
	int basisappx;
	real* basis0;
	real* basis1;
	std::unordered_map<std::string, std::string>* configHash;
	std::unordered_map<std::string, int> bcHash;
	std::unordered_map<int, int> rankHash;

	enum BoundaryConditionTypes {
		RX = -1,
		R0 = 0,
		R1T0 = 1,
		R1T1 = 2,
		R1T2 = 3,
		R1T10 = 4,
		R2T10 = 5,
		R2T20 = 6,
		R3 = 7,
		PERIODIC = 8
	};

	enum BasisApproximation {
		NONE = 0,
		PARTIAL = 1,
		FULL = 2
	};

	real iFilterScale,jFilterScale, kFilterScale;
	RecursiveFilter* iFilter;
	RecursiveFilter* jFilter;
	RecursiveFilter* kFilter;

	ReferenceState* refstate;
	std::string outputPath;

	ErrorData variance;
#pragma acc declare copyin(iDim,jDim,kDim,varDim,kLDim)
#pragma acc declare copyin(iFilterScale,jFilterScale,kFilterScale)
#pragma acc declare copyin(kRankMax)
};

#endif
