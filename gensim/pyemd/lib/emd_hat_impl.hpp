#ifndef EMD_HAT_IMPL_HPP
#define EMD_HAT_IMPL_HPP

//=======================================================================================
// Implementation stuff
//=======================================================================================

#include "min_cost_flow.hpp"
#include <set>
#include <limits>
#include <cassert>
#include <algorithm>
#include <cmath>

template<typename NUM_T>
void fillFWithZeros(std::vector< std::vector<NUM_T> >& F) {
    for (NODE_T i= 0; i<F.size(); ++i) {
        for (NODE_T j= 0; j<F[i].size(); ++j) {
            F[i][j]= 0;
        }
    }
}
        
// Forward declarations 
template<typename NUM_T, FLOW_TYPE_T FLOW_TYPE> struct emd_hat_impl;

template<typename NUM_T,FLOW_TYPE_T FLOW_TYPE>
NUM_T emd_hat_gd_metric<NUM_T,FLOW_TYPE>::operator()(const std::vector<NUM_T>& Pc, const std::vector<NUM_T>& Qc,
                                                     const std::vector< std::vector<NUM_T> >& C,
                                                     NUM_T extra_mass_penalty,
                                                     std::vector< std::vector<NUM_T> >* F) {

    if (FLOW_TYPE!=NO_FLOW) fillFWithZeros(*F);
        
    assert( (F!=NULL) || (FLOW_TYPE==NO_FLOW) );
    
    std::vector<NUM_T> P= Pc;
    std::vector<NUM_T> Q= Qc;
    
    // Assuming metric property we can pre-flow 0-cost edges
    {for (NODE_T i=0; i<P.size(); ++i) {
            if (P[i]<Q[i]) {
                if (FLOW_TYPE!=NO_FLOW) {
                    ((*F)[i][i])= P[i];
                }
                Q[i]-= P[i];
                P[i]= 0;
            } else {
                if (FLOW_TYPE!=NO_FLOW) {
                    ((*F)[i][i])= Q[i];
                }
                P[i]-= Q[i];
                Q[i]= 0;
            }
    }}

    return emd_hat_impl<NUM_T,FLOW_TYPE>()(Pc,Qc,P,Q,C,extra_mass_penalty,F);
    
} // emd_hat_gd_metric

template<typename NUM_T,FLOW_TYPE_T FLOW_TYPE>
NUM_T emd_hat<NUM_T,FLOW_TYPE>::operator()(const std::vector<NUM_T>& P, const std::vector<NUM_T>& Q,
                                           const std::vector< std::vector<NUM_T> >& C,
                                           NUM_T extra_mass_penalty,
                                           std::vector< std::vector<NUM_T> >* F) {

    if (FLOW_TYPE!=NO_FLOW) fillFWithZeros(*F);
    return emd_hat_impl<NUM_T,FLOW_TYPE>()(P,Q,P,Q,C,extra_mass_penalty,F);

} // emd_hat


//-----------------------------------------------------------------------------------------------
// Implementing it for different types
//-----------------------------------------------------------------------------------------------

// Blocking instantiation for a non-overloaded template param
template<typename NUM_T, FLOW_TYPE_T FLOW_TYPE>
struct emd_hat_impl {
        
}; // emd_hat_impl


//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Main implementation
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
template<typename NUM_T, FLOW_TYPE_T FLOW_TYPE>
struct emd_hat_impl_integral_types {

    NUM_T operator()(
        const std::vector<NUM_T>& POrig, const std::vector<NUM_T>& QOrig,
        const std::vector<NUM_T>& Pc, const std::vector<NUM_T>& Qc,
        const std::vector< std::vector<NUM_T> >& Cc,
        NUM_T extra_mass_penalty,
        std::vector< std::vector<NUM_T> >* F) {

    //-------------------------------------------------------
    NODE_T N= Pc.size();
    assert(Qc.size()==N);

    // Ensuring that the supplier - P, have more mass.
    std::vector<NUM_T> P;
    std::vector<NUM_T> Q;
    std::vector< std::vector<NUM_T> > C(Cc);
    NUM_T abs_diff_sum_P_sum_Q;
    NUM_T sum_P= 0;
    NUM_T sum_Q= 0;
    {for (NODE_T i=0; i<N; ++i) sum_P+= Pc[i];}
    {for (NODE_T i=0; i<N; ++i) sum_Q+= Qc[i];}
    bool needToSwapFlow= false;
    if (sum_Q>sum_P) {
        needToSwapFlow= true;
        P= Qc;
        Q= Pc;
        // transpose C
        for (NODE_T i=0; i<N; ++i) {
            for (NODE_T j=0; j<N; ++j) {
                C[i][j]= Cc[j][i];
            }
        }
        abs_diff_sum_P_sum_Q= sum_Q-sum_P;
    } else {
        P= Pc;
        Q= Qc;
        abs_diff_sum_P_sum_Q= sum_P-sum_Q;
    }
    //if (needToSwapFlow) cout << "needToSwapFlow" << endl;
    
    // creating the b vector that contains all vertexes
    std::vector<NUM_T> b(2*N+2);
    const NODE_T THRESHOLD_NODE= 2*N;
    const NODE_T ARTIFICIAL_NODE= 2*N+1; // need to be last !
    {for (NODE_T i=0; i<N; ++i) {
        b[i]= P[i];
    }}
    {for (NODE_T i=N; i<2*N; ++i) {
        b[i]= (Q[i-N]);
    }}
    
    // remark*) I put here a deficit of the extra mass, as mass that flows to the threshold node
    // can be absorbed from all sources with cost zero (this is in reverse order from the paper,
    // where incoming edges to the threshold node had the cost of the threshold and outgoing
    // edges had the cost of zero)
    // This also makes sum of b zero.
    b[THRESHOLD_NODE]= -abs_diff_sum_P_sum_Q; 
    b[ARTIFICIAL_NODE]= 0;
    //-------------------------------------------------------
    
    //-------------------------------------------------------
    NUM_T maxC= 0;
    {for (NODE_T i=0; i<N; ++i) {
        {for (NODE_T j=0; j<N; ++j) {
                assert(C[i][j]>=0);
                if ( C[i][j]>maxC ) maxC= C[i][j];
        }}
    }}
    if (extra_mass_penalty==-1) extra_mass_penalty= maxC;
    //-------------------------------------------------------
   
    
    //=============================================================
    std::set< NODE_T > sources_that_flow_not_only_to_thresh; 
    std::set< NODE_T > sinks_that_get_flow_not_only_from_thresh; 
    NUM_T pre_flow_cost= 0;
    //=============================================================


    
    //=============================================================
    // regular edges between sinks and sources without threshold edges
    std::vector< std::list< edge<NUM_T> > > c(b.size());
    {for (NODE_T i=0; i<N; ++i) {
        if (b[i]==0) continue;
        {for (NODE_T j=0; j<N; ++j) {
            if (b[j+N]==0) continue;
            if (C[i][j]==maxC) continue;
            c[i].push_back( edge<NUM_T>(j+N , C[i][j]) );
            }} // j
    }}// i

     // checking which are not isolated
     {for (NODE_T i=0; i<N; ++i) {
        if (b[i]==0) continue;
        {for (NODE_T j=0; j<N; ++j) {
            if (b[j+N]==0) continue;
            if (C[i][j]==maxC) continue;
            sources_that_flow_not_only_to_thresh.insert(i);
            sinks_that_get_flow_not_only_from_thresh.insert(j+N);
        }} // j
    }}// i

    // converting all sinks to negative
      {for (NODE_T i=N; i<2*N; ++i) {
              b[i]= -b[i];
    }}
    
     
    // add edges from/to threshold node,
    // note that costs are reversed to the paper (see also remark* above)
    // It is important that it will be this way because of remark* above.
    {for (NODE_T i=0; i<N; ++i) {
            c[i].push_back( edge<NUM_T>(THRESHOLD_NODE, 0) );
    }}
    {for (NODE_T j=0; j<N; ++j) {
            c[THRESHOLD_NODE].push_back( edge<NUM_T>(j+N, maxC) );
    }} 
    
    // artificial arcs - Note the restriction that only one edge i,j is artificial so I ignore it...
    {for (NODE_T i=0; i<ARTIFICIAL_NODE; ++i) {
            c[i].push_back( edge<NUM_T>(ARTIFICIAL_NODE, maxC + 1 ) );
            c[ARTIFICIAL_NODE].push_back( edge<NUM_T>(i, maxC + 1 ) );
    }}
    //=============================================================

    
    

    
    //====================================================    
    // remove nodes with supply demand of 0
    // and vertexes that are connected only to the
    // threshold vertex
    //====================================================    
    NODE_T current_node_name= 0;
    // Note here it should be vector<int> and not vector<NODE_T>
    // as I'm using -1 as a special flag !!!
    const int REMOVE_NODE_FLAG= -1;
    std::vector<int> nodes_new_names(b.size(),REMOVE_NODE_FLAG);
    std::vector<int> nodes_old_names;
    nodes_old_names.reserve(b.size());
    {for (NODE_T i=0; i<N*2; ++i) {
            if (b[i]!=0) {
             if (sources_that_flow_not_only_to_thresh.find(i)!=sources_that_flow_not_only_to_thresh.end()|| 
                sinks_that_get_flow_not_only_from_thresh.find(i)!=sinks_that_get_flow_not_only_from_thresh.end()) {
                nodes_new_names[i]= current_node_name;
                nodes_old_names.push_back(i);
                ++current_node_name;
                } else {
                  if (i>=N) { // sink
                      pre_flow_cost-= (b[i]*maxC);
                  }
                  b[THRESHOLD_NODE]+= b[i]; // add mass(i<N) or deficit (i>=N)
             } 
            }
    }} //i
    nodes_new_names[THRESHOLD_NODE]= current_node_name;
    nodes_old_names.push_back(THRESHOLD_NODE);
    ++current_node_name;
    nodes_new_names[ARTIFICIAL_NODE]= current_node_name;
    nodes_old_names.push_back(ARTIFICIAL_NODE);
    ++current_node_name;

    std::vector<NUM_T> bb(current_node_name);
    NODE_T j=0;
    {for (NODE_T i=0; i<b.size(); ++i) {
        if (nodes_new_names[i]!=REMOVE_NODE_FLAG) {
            bb[j]= b[i];
            ++j;
        }
    }}
        
    std::vector< std::list< edge<NUM_T> > > cc(bb.size());
    {for (NODE_T i=0; i<c.size(); ++i) {
        if (nodes_new_names[i]==REMOVE_NODE_FLAG) continue;
        {for (typename std::list< edge<NUM_T> >::const_iterator it= c[i].begin(); it!=c[i].end(); ++it) {
            if ( nodes_new_names[it->_to]!=REMOVE_NODE_FLAG) {
                cc[ nodes_new_names[i] ].push_back( edge<NUM_T>( nodes_new_names[it->_to], it->_cost ) );
            }
        }}
    }}
    //====================================================    

    #ifndef NDEBUG
    NUM_T DEBUG_sum_bb= 0;
    for (NODE_T i=0; i<bb.size(); ++i) DEBUG_sum_bb+= bb[i];
    assert(DEBUG_sum_bb==0);
    #endif

    //-------------------------------------------------------
    min_cost_flow<NUM_T> mcf;
        
    NUM_T my_dist;
    
    std::vector< std::list<  edge0<NUM_T>  > > flows(bb.size());

    //std::cout << bb.size() << std::endl;
    //std::cout << cc.size() << std::endl;

    //tictoc timer;
    //timer.tic();
    NUM_T mcf_dist= mcf(bb,cc,flows);
    //timer.toc();
    //std::cout << "min_cost_flow time== " << timer.totalTimeSec() << std::endl;

    if (FLOW_TYPE!=NO_FLOW) {
        for (NODE_T new_name_from=0; new_name_from<flows.size(); ++new_name_from) {
            for (typename std::list<  edge0<NUM_T>  >::const_iterator it= flows[new_name_from].begin(); it!=flows[new_name_from].end(); ++it) {
                if (new_name_from==nodes_new_names[THRESHOLD_NODE]||it->_to==nodes_new_names[THRESHOLD_NODE]) continue;
                NODE_T i,j;
                NUM_T flow= it->_flow;
                bool reverseEdge= it->_to<new_name_from;
                if (!reverseEdge) {
                    i= nodes_old_names[new_name_from];
                    j= nodes_old_names[it->_to]-N; 
                } else {
                    i= nodes_old_names[it->_to];
                    j= nodes_old_names[new_name_from]-N;
                }
                if (flow!=0&&new_name_from!=nodes_new_names[THRESHOLD_NODE]&&it->_to!=nodes_new_names[THRESHOLD_NODE]) {
                    assert(i<N&&j<N);
                    if (needToSwapFlow) std::swap(i,j);
                    if (!reverseEdge) {
                        (*F)[i][j]+= flow;
                    } else {
                        (*F)[i][j]-= flow;
                    }
                }
            }
        }
    }
    
    if (FLOW_TYPE==WITHOUT_EXTRA_MASS_FLOW) transform_flow_to_regular(*F,POrig,QOrig);
    
    my_dist=
        pre_flow_cost + // pre-flowing on cases where it was possible
        mcf_dist + // solution of the transportation problem
        (abs_diff_sum_P_sum_Q*extra_mass_penalty); // emd-hat extra mass penalty

    
    return my_dist;
    //-------------------------------------------------------
    
} // emd_hat_impl_integral_types (main implementation) operator()
};
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


//----------------------------------------------------------------------------------------
// integral types
//----------------------------------------------------------------------------------------
template<FLOW_TYPE_T FLOW_TYPE>
struct emd_hat_impl<int,FLOW_TYPE> {

    typedef int NUM_T;
    
    NUM_T operator()(
        const std::vector<NUM_T>& POrig, const std::vector<NUM_T>& QOrig,
        const std::vector<NUM_T>& P, const std::vector<NUM_T>& Q,
        const std::vector< std::vector<NUM_T> >& C,
        NUM_T extra_mass_penalty,
        std::vector< std::vector<NUM_T> >* F) {
        return emd_hat_impl_integral_types<NUM_T,FLOW_TYPE>()(POrig,QOrig,P,Q,C,extra_mass_penalty,F);
    }
    
}; // emd_hat_impl<int>

template<FLOW_TYPE_T FLOW_TYPE>
struct emd_hat_impl<long int,FLOW_TYPE> {

    typedef long int NUM_T;
        
    NUM_T operator()(
        const std::vector<NUM_T>& POrig, const std::vector<NUM_T>& QOrig,
        const std::vector<NUM_T>& P, const std::vector<NUM_T>& Q,
        const std::vector< std::vector<NUM_T> >& C,
        NUM_T extra_mass_penalty,
        std::vector< std::vector<NUM_T> >* F) {
        return emd_hat_impl_integral_types<NUM_T,FLOW_TYPE>()(POrig,QOrig,P,Q,C,extra_mass_penalty,F);
    }

    
}; // emd_hat_impl<long int>

template<FLOW_TYPE_T FLOW_TYPE>
struct emd_hat_impl<long long int,FLOW_TYPE> {

    typedef long long int NUM_T;
    
    NUM_T operator()(
        const std::vector<NUM_T>& POrig, const std::vector<NUM_T>& QOrig,
        const std::vector<NUM_T>& P, const std::vector<NUM_T>& Q,
        const std::vector< std::vector<NUM_T> >& C,
        NUM_T extra_mass_penalty,
        std::vector< std::vector<NUM_T> >* F) {
        return emd_hat_impl_integral_types<NUM_T,FLOW_TYPE>()(POrig,QOrig,P,Q,C,extra_mass_penalty,F);
    }
    
}; // emd_hat_impl<long long int>
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// floating types
//----------------------------------------------------------------------------------------
template<FLOW_TYPE_T FLOW_TYPE>
struct emd_hat_impl<double,FLOW_TYPE> {

    typedef double NUM_T;
    typedef long long int CONVERT_TO_T;
        
    NUM_T operator()(
        const std::vector<NUM_T>& POrig, const std::vector<NUM_T>& QOrig,
        const std::vector<NUM_T>& P, const std::vector<NUM_T>& Q,
        const std::vector< std::vector<NUM_T> >& C,
        NUM_T extra_mass_penalty,
        std::vector< std::vector<NUM_T> >* F) {
        
    // TODO: static assert
    assert(sizeof(CONVERT_TO_T)>=8);
    
    // This condition should hold:
    // ( 2^(sizeof(CONVERT_TO_T*8)) >= ( MULT_FACTOR^2 )
    // Note that it can be problematic to check it because
    // of overflow problems. I simply checked it with Linux calc
    // which has arbitrary precision.
    const double MULT_FACTOR= 1000000; 

    // Constructing the input
    const NODE_T N= P.size();
    std::vector<CONVERT_TO_T> iPOrig(N);
    std::vector<CONVERT_TO_T> iQOrig(N);
    std::vector<CONVERT_TO_T> iP(N);
    std::vector<CONVERT_TO_T> iQ(N);
    std::vector< std::vector<CONVERT_TO_T> > iC(N, std::vector<CONVERT_TO_T>(N) );
    std::vector< std::vector<CONVERT_TO_T> > iF(N, std::vector<CONVERT_TO_T>(N) );

    // Converting to CONVERT_TO_T
    double sumP= 0.0;
    double sumQ= 0.0;
    double maxC= C[0][0];
    for (NODE_T i= 0; i<N; ++i) {
        sumP+= POrig[i];
        sumQ+= QOrig[i];
        for (NODE_T j= 0; j<N; ++j) {
            if (C[i][j]>maxC) maxC= C[i][j];
        }
    }
    double minSum= std::min(sumP,sumQ);
    double maxSum= std::max(sumP,sumQ);
    double PQnormFactor= MULT_FACTOR/maxSum;
    double CnormFactor= MULT_FACTOR/maxC;
    for (NODE_T i= 0; i<N; ++i) {
        iPOrig[i]= static_cast<CONVERT_TO_T>(floor(POrig[i]*PQnormFactor+0.5));
        iQOrig[i]= static_cast<CONVERT_TO_T>(floor(QOrig[i]*PQnormFactor+0.5));
        iP[i]= static_cast<CONVERT_TO_T>(floor(P[i]*PQnormFactor+0.5));
        iQ[i]= static_cast<CONVERT_TO_T>(floor(Q[i]*PQnormFactor+0.5));
        for (NODE_T j= 0; j<N; ++j) {
            iC[i][j]= static_cast<CONVERT_TO_T>(floor(C[i][j]*CnormFactor+0.5));
            if (FLOW_TYPE!=NO_FLOW) {
                iF[i][j]= static_cast<CONVERT_TO_T>(floor(((*F)[i][j])*PQnormFactor+0.5));
            }
        }
    }

    // computing distance without extra mass penalty
    double dist= emd_hat_impl<CONVERT_TO_T,FLOW_TYPE>()(iPOrig,iQOrig,iP,iQ,iC,0,&iF);
    // unnormalize
    dist= dist/PQnormFactor;
    dist= dist/CnormFactor;
    
    // adding extra mass penalty
    if (extra_mass_penalty==-1) extra_mass_penalty= maxC;
    dist+= (maxSum-minSum)*extra_mass_penalty;
        
    // converting flow to double
    if (FLOW_TYPE!=NO_FLOW) {
        for (NODE_T i= 0; i<N; ++i) {
            for (NODE_T j= 0; j<N; ++j) {
                (*F)[i][j]= (iF[i][j]/PQnormFactor);
            }
        }
    }
    
    return dist;
    }
    
}; // emd_hat_impl<double>
//----------------------------------------------------------------------------------------
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
