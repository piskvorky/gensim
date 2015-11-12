#ifndef EMD_HAT_HPP
#define EMD_HAT_HPP

#include <vector>
#include "EMD_DEFS.hpp"
#include "flow_utils.hpp"

/// Fastest version of EMD. Also, in my experience metric ground distance yields better
/// performance.
///
/// Required params:
/// P,Q - Two histograms of size N
/// C - The NxN matrix of the ground distance between bins of P and Q. Must be a metric. I
///     recommend it to be a thresholded metric (which is also a metric, see ICCV paper).
///
/// Optional params:
/// extra_mass_penalty - The penalty for extra mass - If you want the
///                     resulting distance to be a metric, it should be
///                     at least half the diameter of the space (maximum
///                     possible distance between any two points). If you
///                     want partial matching you can set it to zero (but
///                     then the resulting distance is not guaranteed to be a metric).
///                     Default value is -1 which means 1*max_element_in_C
/// F - *F is filled with flows or nothing happens to F. See template param FLOW_TYPE.
///     Note that EMD and EMD-HAT does not necessarily have a unique flow solution.
///     We assume *F is already allocated and has enough space and is initialized to zeros.
///     See also flow_utils.hpp file for flow-related utils.
///     Default value: NULL and then FLOW_TYPE must be NO_FLOW.
///
/// Required template params:
/// NUM_T - the type of the histogram bins count (should be one of: int, long int, long long int, double)
///
/// Optional template params:
/// FLOW_TYPE == NO_FLOW - does nothing with the given F.
///           == WITHOUT_TRANSHIPMENT_FLOW - fills F with the flows between bins connected
///              with edges smaller than max(C).
///           == WITHOUT_EXTRA_MASS_FLOW - fills F with the flows between all bins, except the flow
///              to the extra mass bin.
///           Note that if F is the default NULL then FLOW_TYPE must be NO_FLOW.

template<typename NUM_T, FLOW_TYPE_T FLOW_TYPE= NO_FLOW>
struct emd_hat_gd_metric {
    NUM_T operator()(const std::vector<NUM_T>& P, const std::vector<NUM_T>& Q,
                     const std::vector< std::vector<NUM_T> >& C,
                     NUM_T extra_mass_penalty= -1,
                     std::vector< std::vector<NUM_T> >* F= NULL);
};

/// Same as emd_hat_gd_metric, but does not assume metric property for the ground distance (C).
/// Note that C should still be symmetric and non-negative!
template<typename NUM_T, FLOW_TYPE_T FLOW_TYPE= NO_FLOW>
struct emd_hat {
    NUM_T operator()(const std::vector<NUM_T>& P, const std::vector<NUM_T>& Q,
                     const std::vector< std::vector<NUM_T> >& C,
                     NUM_T extra_mass_penalty= -1,
                     std::vector< std::vector<NUM_T> >* F= NULL);

};


/// =========================================================================
/// 02/27/2014 - Added by Will Mayner <wmayner@gmail.com>
/// -------------------------------------------------------------------------
/// Instantiate the templates with various types to easily import into Cython
emd_hat_gd_metric<int> emd_hat_gd_metric_int;
emd_hat_gd_metric<double> emd_hat_gd_metric_double;
emd_hat_gd_metric<long int> emd_hat_gd_metric_long_int;
emd_hat_gd_metric<long long int> emd_hat_gd_metric_long_long_int;
emd_hat<int> emd_hat_int;
emd_hat<double> emd_hat_double;
emd_hat<long int> emd_hat_long_int;
emd_hat<long long int> emd_hat_long_long_int;
/// =========================================================================


#include "emd_hat_impl.hpp"

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

