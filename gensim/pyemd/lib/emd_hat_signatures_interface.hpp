#ifndef _EMD_HAT_SIGNATURE_INTERFACE_HXX
#define _EMD_HAT_SIGNATURE_INTERFACE_HXX

#include "EMD_DEFS.hpp"
#include "emd_hat.hpp"

//=============================================================================
// This interface is similar to Rubner's interface. See:
// http://www.cs.duke.edu/~tomasi/software/emd.htm
// With the following changes;
// 1. Weights of signature should be of type NUM_T (see emd_hat.hpp)
// 2. Return value of the distance function (func) should be of type NUM_T 
// 3. Return value of the emd_hat_signature_interface function is NUM_T 
// 4. The function does not return a flow (I may add this in future, if needed)
// 5. The function also gets the penalty for extra mass - if you want metric property
//    should be at least half the diameter of the space (maximum possible distance
//    between any two points). In Rubner's code this is implicitly 0. 
// 6. The result is not normalized with the flow.
//
// To get the same results as Rubner's code you should set extra_mass_penalty to 0,
// and divide by the minimum of the sum of the two signature's weights. However, I
// suggest not to do this as you lose the metric property and more importantly, in my
// experience the performance is better with emd_hat. for more on the difference
// between emd and emd_hat, see the paper:
//  A Linear Time Histogram Metric for Improved SIFT Matching
//  Ofir Pele, Michael Werman
//  ECCV 2008
//
// To get shorter running time, set the ground distance function (func) to
// be a thresholded distance. For example: min( L2, T ). Where T is some threshold.
// Note that the running time is shorter with smaller T values. Note also that
// thresholding the distance will probably increase accuracy. Finally, a thresholded
// metric is also a metric. See paper:
//  Fast and Robust Earth Mover's Distances
//	Ofir Pele, Michael Werman
//  ICCV 2009
//
// If you use this code, please cite the papers.
//=============================================================================

/*****************************************************************************/
/* feature_tt SHOULD BE MODIFIED BY THE USER TO REFLECT THE FEATURE TYPE      */
typedef double feature_tt;
/*****************************************************************************/

template<typename NUM_T>
struct signature_tt {
    int n;                /* Number of features in the signature */
    feature_tt* Features; /* Pointer to the features vector */
    NUM_T* Weights;         /* Pointer to the weights of the features (Changed from Rubner's)*/
};

/// Similar to Rubner's emd interface.
/// extra_mass_penalty - it's alpha*maxD_ij in my ECCV paper. If you want metric property
///                      should be at least half the diameter of the space (maximum possible distance
///                      between any two points). In Rubner's code this is implicitly 0.
///                      Default value is -1 which means 1*max_distance_between_bins_of_signatures
template<typename NUM_T>
NUM_T emd_hat_signature_interface(signature_tt<NUM_T>* Signature1, signature_tt<NUM_T>* Signature2,
                                  NUM_T (*func)(feature_tt*, feature_tt*),
                                  NUM_T extra_mass_penalty) {
    
    std::vector<NUM_T> P(Signature1->n + Signature2->n , 0);
    std::vector<NUM_T> Q(Signature1->n + Signature2->n , 0); 
    for (int i=0; i<Signature1->n; ++i) {
        P[i]= Signature1->Weights[i];
    }
    for (int j=0; j<Signature2->n; ++j) {
        Q[j+Signature1->n]= Signature2->Weights[j];
    }
    
    std::vector< std::vector<NUM_T> > C(P.size(), std::vector<NUM_T>(P.size(), 0) );
    {for (int i=0; i<Signature1->n; ++i) {
        {for (int j=0; j<Signature2->n; ++j) {
            NUM_T dist= func( (Signature1->Features+i) , (Signature2->Features+j) );
            assert(dist>=0);
            C[i][j+Signature1->n]= dist;
            C[j+Signature1->n][i]= dist;
        }}
    }}

    return emd_hat<NUM_T,NO_FLOW>()(P,Q,C, extra_mass_penalty);

} // emd_hat_signature_interface

#endif

// Copyright (c) 2009-2012, Ofir Pele
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met: 
//    * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//    * Neither the name of the The Hebrew University of Jerusalem nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

