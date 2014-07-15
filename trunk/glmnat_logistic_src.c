/*
 * AUTHOR : Tom Michoel, The Roslin Institute
 *          tom.michoel@roslin.ed.ac.uk
 *          http://lab.michoel.info
 *
 * REFERENCE: Michoel T. Natural coordinate descent algorithm for 
 *  L1-penalised regression in generalised linear models. arXiv:1405.4225
 *
 * LICENSE: GNU GPL v2
 *
 * INSTALLATION: Compile on Matlab prompt with
 *     mex -I<GLIBDIR>/include/glib-2.0 -I<GLIBDIR>/lib/glib-2.0/include/ -lglib-2.0 glmnat_logistic_src.c
 *  where <GLIBDIR> is the location of glib, e.g. on my mac
 *  <GLIBDIR> = /usr/local/Cellar/glib/2.34.1/
 *  See test_glmnat_logistic.m for usage example.
 */

#include "mex.h"
#include "matrix.h"
#include "tgmath.h"
#include <glib.h>

/*
 * Return sign of a double
 */
static inline int sgn(double val) {
  if (val < 0) return -1;
  if (val==0) return 0;
  return 1;
}

/*
 * Update natural parameter of the GLM + some functions of it
 */
void updateGLM (double *eta, double *etaFunc1, double *etaFunc2, double *X, 
        double bdiff, int j, int n){
    double val, expval1, expval2;
    for (int i=0; i<n; i++){
         val = eta[i] + bdiff*X[i+n*j];
         expval1 = exp(-val);
         expval2 = 1 / (n*(1+expval1));
         eta[i] = val;
         etaFunc1[i] = expval2;
         etaFunc2[i] = expval1*expval2;
    }
}

/*
 * Find unconstrained minimizer of Legendre transform of potential U for coordinate j
 */
void minLegendreU(double *w0, double *X, double *eta, double *etaFunc1, 
        double betaj, int j, int n) {
    double val = 0.0, xval;
    if (betaj==0.0){
        for (int i=0; i<n; i++){
            val += X[i+n*j] * etaFunc1[i];
        }
    } else {
        for (int i=0; i<n; i++){
            xval = X[i+n*j];
            val += xval / (n*(1+exp(-eta[i]+xval*betaj)));
        }
    }
    w0[j] = val;
}

/*
 * First and second derivative of potential U for coordindate j
 */
void derivU (double *X, double *eta, double *etaFunc1, double *etaFunc2, 
        double *returnVal, int n, int j){
    double val1=0.0, val2=0.0;
    for (int i=0; i<n; i++){
        val1 += X[i+n*j] * etaFunc1[i];
        val2 += n * pow(X[i+n*j],2) * etaFunc1[i] * etaFunc2[i];
    }
    returnVal[0] = val1;
    returnVal[1] = val2;
}

/*
 * Derivative of Legendre transform of potential U at its constrained
 * minimiser
 */
double constrainedMinLegendreU(double *X, double *eta, double *etaFunc1, 
        double *etaFunc2, double betaj, double ww, int j, int n){
    double betanew;
    double *returnVal;
    returnVal = mxGetPr( mxCreateDoubleMatrix(2, 1, mxREAL) ); 
    derivU(X, eta, etaFunc1, etaFunc2, returnVal, n, j);
    betanew = betaj + (ww - returnVal[0])/returnVal[1];
    return betanew;
}


/*
 * Update coordinate j
 */
double coordinateUpdate(double *w0, double *X, double *eta, double *etaFunc1, 
        double *etaFunc2, double *w, double *beta, double mu, int j, int n) {
    double changed = 0.0;
    double bj = beta[j], wj = w[j], ww, b;
    
    // update unpenalised minimum
    minLegendreU(w0, X, eta, etaFunc1, bj, j, n);
    
    // soft-thresholding operation
    if ( fabs(wj-w0[j]) > mu ) {
        ww = wj-sgn(wj-w0[j])*mu;
        b = constrainedMinLegendreU(X, eta, etaFunc1, etaFunc2, bj, ww, j, n);
    } else
        b = 0.0;
    
    // update if necessary
    if ( b!=bj ) {
        changed = fabs(b-bj);
        updateGLM(eta, etaFunc1, etaFunc2, X, b-bj, j, n);
        beta[j] = b;
    }

    return changed;
}


/*
 * Complete coordinate descent cycle along all coordinates
 */
GSList* cycleComplete (double *w0, double *X, GSList* activeSet, 
        double *eta, double *etaFunc1, double *etaFunc2, double *w, 
        double *beta, double mu, int n, int p, double *normdiffPtr){

    // convergence parameters
    double changedj, normdiff = 0.0;
    
    // reset active set
    activeSet = NULL;
    
    // 1st coefficient is unpenalised intercept
    changedj = coordinateUpdate(w0, X, eta, etaFunc1, etaFunc2, w, beta,
            0.0, 0, n);
    normdiff = fmax(normdiff,changedj);
    
    // now the penalised coefficients
    for (int j=1; j<p; j++){
        changedj = coordinateUpdate(w0, X, eta, etaFunc1, etaFunc2, w,
                beta, mu, j, n);
        normdiff = fmax(normdiff,changedj);
        if (beta[j] != 0.0){
            activeSet = g_slist_append(activeSet, (void *)j);
        }
    }
    
    // store maximum difference of new and old estimate
    *normdiffPtr = normdiff;
    
    // return activeSet
    return activeSet;
}

/*
 * Coordinate descent cycle along active coordinates only
 */
void cycleActive (double *w0, double *X, GSList* activeSet, double *eta, double *etaFunc1, 
        double *etaFunc2, double *w, double *beta, double mu, int n, int p,
        double *normdiffPtr){

    // convergence parameters
    double changedj, normdiff = 0.0;
    int j;    
    GSList *itj = NULL;
    
    // 1st coefficient is unpenalised intercept and is always in active set
    if (beta[0]!=0){
        changedj = coordinateUpdate(w0, X, eta, etaFunc1, etaFunc2, w, beta,
                0.0, 0, n);
        normdiff = fmax(normdiff,changedj);
    }
    
    // now the penalised coefficients in the active set
    for (itj = activeSet; itj; itj = itj->next) {
        j = (int)itj->data;
        changedj = coordinateUpdate(w0, X, eta, etaFunc1, etaFunc2, w,
                beta, mu, j, n);
        normdiff = fmax(normdiff,changedj);
    }
    
    // store maximum difference of new and old estimate
    *normdiffPtr = normdiff;

}

/*
 * Main function
 */ 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    double *X, *w, *museq, *betaMat;  // pointers to input/output arrays
    double *beta, *eta, *etaFunc1, *etaFunc2, *w0, *normdiff, *normdiffActive, tol;
    mwSize n, p, m;     // problem dimensions
    int iter = 0, iterAct = 0, max_iter = 100, iter_all;
    
    X = mxGetPr(prhs[0]); // predictor input matrix including intercept column
    w = mxGetPr(prhs[1]); // predictor-response overlap vector
    museq = mxGetPr(prhs[2]); // column vector of L1 penalty parameters
    tol = *mxGetPr(prhs[3]); // convergence tolerance threshold
    
    n = mxGetM(prhs[0]); // number of samples 
    p = mxGetN(prhs[0]); // number of predictors including intercept
    m = mxGetM(prhs[2]); // number of L1 penalty parameters

    plhs[0] = mxCreateDoubleMatrix(p, m, mxREAL);
    betaMat = mxGetPr(plhs[0]);// output matrix with regression coefficients
    
    w0 = mxGetPr( mxCreateDoubleMatrix(p, 1, mxREAL) );
    w0[0] = 0.5; // vector with unpenalised minima locations
    
    beta = mxGetPr( mxCreateDoubleMatrix(p, 1, mxREAL) ); // current estimates
    eta = mxGetPr( mxCreateDoubleMatrix(n, 1, mxREAL) ); // GLM parameter vector
    etaFunc1 = mxGetPr( mxCreateDoubleMatrix(n, 1, mxREAL) ); // GLM parameter vector function 1
    etaFunc2 = mxGetPr( mxCreateDoubleMatrix(n, 1, mxREAL) ); // GLM parameter vector function 1
    
    normdiff = mxGetPr( mxCreateDoubleMatrix(1, 1, mxREAL) );
    normdiffActive = mxGetPr( mxCreateDoubleMatrix(1, 1, mxREAL) );
    
    // active set of coordinates
    GSList* activeSet = NULL;
    
    // initialise GLM
    updateGLM(eta, etaFunc1, etaFunc2, X, 0.0, 0, n);
    activeSet = cycleComplete (w0, X, activeSet, eta, etaFunc1, etaFunc2, w, beta, 
            museq[0], n, p, normdiff);
    
    for (int k=0; k<m; k++){
        iter = 0;
        do {
            iter++;
            iterAct = 0;
            do {
                iterAct++;
                cycleActive (w0, X, activeSet, eta, etaFunc1, etaFunc2,
                        w, beta, museq[k], n, p, normdiffActive);
            } while (*normdiffActive>tol && iterAct < max_iter);
            if (iterAct>=max_iter){
                printf("Warning :: active set did not converge for mu=%f\n", museq[k]);
            }
            
            activeSet = cycleComplete (w0, X, activeSet, eta, etaFunc1, etaFunc2,
                    w, beta, museq[k], n, p, normdiff);
            
        } while (*normdiff>tol && iter < max_iter);
        if (iter>=max_iter){
            printf("Warning :: did not converge for mu=%f\n", museq[k]);
        }

        // store next column of betaMat
        for (int j=0; j<p; j++){
            betaMat[j+p*k] = beta[j];
        }
    }
    
    
    

}