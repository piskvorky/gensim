#ifndef MIN_COST_FLOW_HXX
#define MIN_COST_FLOW_HXX

#include <vector>
#include <limits>
#include <list>
#include <cassert>
#include <math.h>
#include "EMD_DEFS.hpp"

//------------------------------------------------------------------------------
template<typename NUM_T>
struct edge {
    edge(NODE_T to, NUM_T cost) : _to(to), _cost(cost) {}
    NODE_T _to;
    NUM_T _cost;
};

template<typename NUM_T>
struct edgeCompareByCost {
    bool operator()(const edge<NUM_T>& a, const edge<NUM_T>& b) {
        return a._cost<b._cost;
    }
};

template<typename NUM_T>
struct edge0 {
    edge0(NODE_T to, NUM_T cost, NUM_T flow) : _to(to), _cost(cost), _flow(flow) {}
    NODE_T _to;
    NUM_T _cost;
    NUM_T _flow;
};

template<typename NUM_T>
struct edge1 {
    edge1(NODE_T to, NUM_T reduced_cost) : _to(to), _reduced_cost(reduced_cost) {}
    NODE_T _to;
    NUM_T _reduced_cost;
};

template<typename NUM_T>
struct edge2 {
    edge2(NODE_T to, NUM_T reduced_cost, NUM_T residual_capacity) : _to(to), _reduced_cost(reduced_cost), _residual_capacity(residual_capacity) {}
    NODE_T _to;
    NUM_T _reduced_cost;
    NUM_T _residual_capacity;
};

template<typename NUM_T>
struct edge3 {
    edge3(NODE_T to=0, NUM_T dist=0) : _to(to), _dist(dist) {}
    NODE_T _to;
    NUM_T _dist;
};
//------------------------------------------------------------------------------







//------------------------------------------------------------------------------
template<typename NUM_T>
class min_cost_flow {

    NODE_T _num_nodes;
    std::vector<NODE_T> _nodes_to_Q;

    //tictoc tictoc_shortest_path;
    //tictoc tictoc_while_true;
    //tictoc tmp_tic_toc;
    //tictoc tictoc_all_function;

public:

    // e - supply(positive) and demand(negative).
    // c[i] - edges that goes from node i. first is the second nod
    // x - the flow is returned in it
    NUM_T operator()(std::vector<NUM_T>& e,
                     const std::vector< std::list< edge<NUM_T> > >& c,
                     std::vector< std::list<  edge0<NUM_T>  > >& x) {

        //for (NODE_T i=0; i<e.size(); ++i) cout << e[i]<< " ";
        //cout << endl;
        //tictoc_all_function.tic();
                
        assert(e.size()==c.size());
        assert(x.size()==c.size());
        
        _num_nodes= e.size();
        _nodes_to_Q.resize(_num_nodes);
        
        // init flow
        {for (NODE_T from=0; from<_num_nodes; ++from) {
            {for (typename std::list< edge<NUM_T> >::const_iterator it= c[from].begin(); it!=c[from].end(); ++it) {
                x[from].push_back(  edge0<NUM_T> (it->_to, it->_cost, 0) );
                x[it->_to].push_back(  edge0<NUM_T> (from, -it->_cost,0) );
            }} // it
        }} // from
        
        // reduced costs for forward edges (c[i,j]-pi[i]+pi[j])
        // Note that for forward edges the residual capacity is infinity
        std::vector< std::list< edge1<NUM_T> > > r_cost_forward(_num_nodes);
        {for (NODE_T from=0; from<_num_nodes; ++from) {
            {for (typename std::list<  edge<NUM_T>  >::const_iterator it= c[from].begin(); it!=c[from].end(); ++it) {
                    r_cost_forward[from].push_back( edge1<NUM_T>(it->_to,it->_cost) );
            }}
        }}
        
        // reduced costs and capacity for backward edges (c[j,i]-pi[j]+pi[i])
        // Since the flow at the beginning is 0, the residual capacity is also zero
        std::vector< std::list< edge2<NUM_T> > > r_cost_cap_backward(_num_nodes);
        {for (NODE_T from=0; from<_num_nodes; ++from) {
            {for (typename std::list<  edge<NUM_T>  >::const_iterator it= c[from].begin(); it!=c[from].end(); ++it) {
                    r_cost_cap_backward[ it->_to ].push_back( edge2<NUM_T>(from,-it->_cost,0) );
            }} // it
        }} // from
        
        // Max supply TODO:demand?, given U?, optimization-> min out of demand,supply
        NUM_T U= 0;
        {for (NODE_T i=0; i<_num_nodes; ++i) {
            if (e[i]>U) U= e[i];
        }}
        NUM_T delta= static_cast<NUM_T>(pow(2.0l,ceil(log(static_cast<long double>(U))/log(2.0))));

        

        std::vector< NUM_T > d(_num_nodes);
        std::vector< NODE_T > prev(_num_nodes);
        delta= 1;
        //while (delta>=1) {
        
            // delta-scaling phase
            //cout << "delta==" << delta << endl;
        
        //tictoc_while_true.tic();
        while (true) { //until we break when S or T is empty
                
                NUM_T maxSupply= 0;
                NODE_T k=0;
                for (NODE_T i=0; i<_num_nodes; ++i) {
                    if (e[i]>0) {
                        if (maxSupply<e[i]) {
                            maxSupply= e[i];
                            k= i; 
                        }
                    }
                }
                if (maxSupply==0) break;
                delta= maxSupply;

                NODE_T l;
                //tictoc_shortest_path.tic();
                compute_shortest_path(d,prev, k,r_cost_forward,r_cost_cap_backward , e,l); 
                //tictoc_shortest_path.toc(); 
                

                
                //---------------------------------------------------------------
                // find delta (minimum on the path from k to l)
                //delta= e[k];
                //if (-e[l]<delta) delta= e[k];
                NODE_T to= l;
                do {
                    NODE_T from= prev[to];
                    assert(from!=to);
                                        
                    // residual
                    typename std::list< edge2<NUM_T> >::iterator itccb= r_cost_cap_backward[from].begin();
                    while ( (itccb!=r_cost_cap_backward[from].end()) && (itccb->_to!=to) ) {
                        ++itccb;
                    }
                    if (itccb!=r_cost_cap_backward[from].end()) {
                        if (itccb->_residual_capacity<delta) delta= itccb->_residual_capacity;
                    }
                    
                    to= from;
                } while (to!=k);
                //---------------------------------------------------------------

                //---------------------------------------------------------------
                // augment delta flow from k to l (backwards actually...)
                to= l;
                do {
                    NODE_T from= prev[to];
                    assert(from!=to);
                                        
                    // TODO - might do here O(n) can be done in O(1)
                    typename std::list<  edge0<NUM_T>  >::iterator itx= x[from].begin();
                    while (itx->_to!=to) {
                        ++itx;
                    }
                    itx->_flow+= delta;
                                        
                    // update residual for backward edges
                    typename std::list< edge2<NUM_T> >::iterator itccb= r_cost_cap_backward[to].begin();
                    while ( (itccb!=r_cost_cap_backward[to].end()) && (itccb->_to!=from) ) {
                        ++itccb;
                    }
                    if (itccb!=r_cost_cap_backward[to].end()) {
                        itccb->_residual_capacity+= delta;
                    }
                    itccb= r_cost_cap_backward[from].begin();
                    while ( (itccb!=r_cost_cap_backward[from].end()) && (itccb->_to!=to) ) {
                        ++itccb;
                    }
                    if (itccb!=r_cost_cap_backward[from].end()) {
                        itccb->_residual_capacity-= delta;
                    }

                    // update e
                    e[to]+= delta;
                    e[from]-= delta;
                        
                    to= from;
                } while (to!=k);
                //---------------------------------------------------------------------------------

                
                
            } // while true (until we break when S or T is empty)
            //tictoc_while_true.toc();
            //cout << "while true== " << tictoc_while_true.totalTimeSec() << endl;
            
            //delta= delta/2;
            //} // (delta-scaling phase)
            
            
            // compute distance from x
            //cout << endl << endl;
            NUM_T dist= 0;
            {for (NODE_T from=0; from<_num_nodes; ++from) {
                {for (typename std::list<  edge0<NUM_T>  >::const_iterator it= x[from].begin(); it!=x[from].end(); ++it) {
//                        if (it->_flow!=0) cout << from << "->" << it->_to << ": " << it->_flow << "x" << it->_cost << endl;
                        dist+= (it->_cost*it->_flow);
                }} // it
            }} // from
            
            
            //tictoc_all_function.toc();
            //cout << "operator() time==" << tictoc_all_function.totalTimeSec() << endl;
            //cout << "compute_shortest_path_time==" << tictoc_shortest_path.totalTimeSec() << endl;
            //cout << "tmp_tic_toc== " << tmp_tic_toc.totalTimeSec() << endl;
            return dist;
    } // operator()












private:

    void compute_shortest_path(std::vector< NUM_T >& d,
                               std::vector< NODE_T >& prev,
                               
                               NODE_T from,
                               std::vector< std::list< edge1<NUM_T> > >& cost_forward,
                               std::vector< std::list< edge2<NUM_T> > >& cost_backward,

                               const std::vector<NUM_T>& e,
                               NODE_T& l) {
        
        
        //----------------------------------------------------------------
        // Making heap (all inf except 0, so we are saving comparisons...)
        //----------------------------------------------------------------
        std::vector<  edge3<NUM_T>  > Q(_num_nodes);
        
        Q[0]._to= from;
        _nodes_to_Q[from]= 0;
        Q[0]._dist= 0; 
                
        NODE_T j=1;
        // TODO: both of these into a function?
        {for (NODE_T i=0; i<from; ++i) {
            Q[j]._to= i;
            _nodes_to_Q[i]= j;
            Q[j]._dist= std::numeric_limits<NUM_T>::max();
            ++j;
        }}

        {for (NODE_T i=from+1; i<_num_nodes; ++i) {
            Q[j]._to= i;
            _nodes_to_Q[i]= j;
            Q[j]._dist= std::numeric_limits<NUM_T>::max();
            ++j;
        }}
        //----------------------------------------------------------------
        

        //----------------------------------------------------------------
        // main loop
        //----------------------------------------------------------------
        std::vector<NODE_T> finalNodesFlg(_num_nodes, false);
        do {
            NODE_T u= Q[0]._to;
                        
            d[u]= Q[0]._dist; // final distance
            finalNodesFlg[u]= true;
            if (e[u]<0) {
                l= u;
                break;
            }
            
            heap_remove_first(Q, _nodes_to_Q);
            
            
            // neighbors of u    
            {for (typename std::list< edge1<NUM_T> >::const_iterator it= cost_forward[u].begin(); it!=cost_forward[u].end(); ++it) {
                assert (it->_reduced_cost>=0);
                NUM_T alt= d[u]+it->_reduced_cost;
                NODE_T v= it->_to;
                if ( (_nodes_to_Q[v]<Q.size()) && (alt<Q[_nodes_to_Q[v]]._dist) ) {
                    //cout << "u to v==" << u << " to " << v << "   " << alt << endl;
                    heap_decrease_key(Q,_nodes_to_Q, v,alt);
                    prev[v]= u;
                }
            }} //it
            {for (typename std::list< edge2<NUM_T> >::const_iterator it= cost_backward[u].begin(); it!=cost_backward[u].end(); ++it) {
                if (it->_residual_capacity>0) {
                    assert (it->_reduced_cost>=0);
                    NUM_T alt= d[u]+it->_reduced_cost;
                    NODE_T v= it->_to;
                    if ( (_nodes_to_Q[v]<Q.size()) && (alt<Q[_nodes_to_Q[v]]._dist) )  {
                        //cout << "u to v==" << u << " to " << v << "   " << alt << endl;
                        heap_decrease_key(Q,_nodes_to_Q, v,alt);
                        prev[v]= u;
                    }
                }
            }} //it

        } while (!Q.empty());

        
        //tmp_tic_toc.tic();
        //---------------------------------------------------------------------------------
        // reduced costs for forward edges (c[i,j]-pi[i]+pi[j])
								   {for (NODE_T from=0; from<_num_nodes; ++from) {
                                           {for (typename std::list< edge1<NUM_T> >::iterator it= cost_forward[from].begin();
                 it!=cost_forward[from].end(); ++it) {
                if (finalNodesFlg[from]) {
                    it->_reduced_cost+= d[from] - d[l];
                }
                if (finalNodesFlg[it->_to]) {
                    it->_reduced_cost-= d[it->_to] - d[l];
                }
            } }
        }}
        
        // reduced costs and capacity for backward edges (c[j,i]-pi[j]+pi[i])
								   {for (NODE_T from=0; from<_num_nodes; ++from) {
                                           {  for (typename std::list< edge2<NUM_T> >::iterator it= cost_backward[from].begin();
                 it!=cost_backward[from].end(); ++it) {
                if (finalNodesFlg[from]) {
                    it->_reduced_cost+= d[from] - d[l];
                        }
                if (finalNodesFlg[it->_to]) {
                    it->_reduced_cost-= d[it->_to] - d[l];
                }
                               
            } }// it
        }}
        //---------------------------------------------------------------------------------
        //tmp_tic_toc.toc();
        
        //----------------------------------------------------------------
        
        
    } // compute_shortest_path

    void heap_decrease_key(std::vector< edge3<NUM_T> >& Q, std::vector<NODE_T>& nodes_to_Q,
                           NODE_T v, NUM_T alt) {
        NODE_T i= nodes_to_Q[v];
        Q[i]._dist= alt;
        while (i>0 && Q[PARENT(i)]._dist>Q[i]._dist) {
            swap_heap(Q, nodes_to_Q, i, PARENT(i));
            i= PARENT(i);
        }
    } // heap_decrease_key
    
    void heap_remove_first(std::vector< edge3<NUM_T> >& Q, std::vector<NODE_T>& nodes_to_Q) {
        swap_heap(Q, nodes_to_Q, 0, Q.size()-1);
        Q.pop_back();
        heapify(Q,nodes_to_Q , 0);
    } // heap_remove_first



    void heapify(std::vector< edge3<NUM_T> >& Q, std::vector<NODE_T>& nodes_to_Q,
                 NODE_T i) {

        do {
            // TODO: change to loop
            NODE_T l= LEFT(i);
            NODE_T r= RIGHT(i);
            NODE_T smallest;
            if ( (l<Q.size()) && (Q[l]._dist<Q[i]._dist) ) {
                smallest= l;
            } else {
                smallest= i;
            }
            if ( (r<Q.size())&& (Q[r]._dist<Q[smallest]._dist) ) {
                smallest= r;
            }

            if (smallest==i) return;

            swap_heap(Q, nodes_to_Q, i, smallest);
            i= smallest;
            
        } while (true);
        
    } // end heapify




    void swap_heap( std::vector< edge3<NUM_T> >& Q, std::vector<NODE_T>& nodes_to_Q, NODE_T i, NODE_T j) {
        edge3<NUM_T> tmp= Q[i];
        Q[i]= Q[j];
        Q[j]= tmp;
        nodes_to_Q[ Q[j]._to ]= j;
        nodes_to_Q[ Q[i]._to ]= i;
    } // swap_heapify
    
    NODE_T LEFT(NODE_T i) {
        return 2*(i+1)-1;
    } 

    NODE_T RIGHT(NODE_T i) {
        return 2*(i+1); // 2*(i+1)+1-1
    }

    NODE_T PARENT(NODE_T i) {
        return (i-1)/2;
    }
    
}; // end min_cost_flow


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





