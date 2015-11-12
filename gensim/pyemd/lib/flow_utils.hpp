#ifndef FLOW_UTILS_HPP
#define FLOW_UTILS_HPP

#include "EMD_DEFS.hpp"
#include <vector>
#include <cassert>

enum FLOW_TYPE_T {
    NO_FLOW= 0,
    WITHOUT_TRANSHIPMENT_FLOW,
    WITHOUT_EXTRA_MASS_FLOW
};

/// returns the flow from/to transhipment vertex given flow F which was computed using
/// FLOW_TYPE_T of kind WITHOUT_TRANSHIPMENT_FLOW.
template<typename NUM_T>
void return_flow_from_to_transhipment_vertex(const std::vector< std::vector<NUM_T> >& F,
                                             const std::vector<NUM_T>& P,
                                             const std::vector<NUM_T>& Q,
                                             std::vector<NUM_T>& flow_from_P_to_transhipment,
                                             std::vector<NUM_T>& flow_from_transhipment_to_Q) {

    flow_from_P_to_transhipment= P;
    flow_from_transhipment_to_Q= Q;
    for (NODE_T i= 0; i<P.size(); ++i) {
        for (NODE_T j= 0; j<P.size(); ++j) {
            flow_from_P_to_transhipment[i]-= F[i][j];
            flow_from_transhipment_to_Q[j]-= F[i][j];
        }
    }

} // return_flow_from_to_transhipment_vertex


/// Transforms the given flow F which was computed using FLOW_TYPE_T of kind WITHOUT_TRANSHIPMENT_FLOW,
/// to a flow which can be computed using WITHOUT_EXTRA_MASS_FLOW. If you want the flow to the extra mass,
/// you can use return_flow_from_to_transhipment_vertex on the returned F.
template<typename NUM_T>
void transform_flow_to_regular(std::vector< std::vector<NUM_T> >& F,
                               const std::vector<NUM_T>& P,
                               const std::vector<NUM_T>& Q) {

    const NODE_T N= P.size();
    std::vector<NUM_T> flow_from_P_to_transhipment(N);
    std::vector<NUM_T> flow_from_transhipment_to_Q(N);
    return_flow_from_to_transhipment_vertex(F,P,Q,
                                            flow_from_P_to_transhipment,
                                            flow_from_transhipment_to_Q);
    
    NODE_T i= 0;
    NODE_T j= 0;
    while( true ) {

        while (i<N&&flow_from_P_to_transhipment[i]==0) ++i;
        while (j<N&&flow_from_transhipment_to_Q[j]==0) ++j;
        if (i==N||j==N) break;
        
        if (flow_from_P_to_transhipment[i]<flow_from_transhipment_to_Q[j]) {
            F[i][j]+= flow_from_P_to_transhipment[i];
            flow_from_transhipment_to_Q[j]-= flow_from_P_to_transhipment[i];
            flow_from_P_to_transhipment[i]= 0;
        } else {
            F[i][j]+= flow_from_transhipment_to_Q[j];
            flow_from_P_to_transhipment[i]-= flow_from_transhipment_to_Q[j];
            flow_from_transhipment_to_Q[j]= 0;
        }

    }

} // transform_flow_to_regular



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

