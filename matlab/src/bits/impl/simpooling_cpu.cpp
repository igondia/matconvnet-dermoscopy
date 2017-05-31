// @file simpooling_cpu.cpp
// @brief Simmetry Pooling block implementation (cpu)
// @author Iván González Díaz

#include "simpooling.hpp"
#include "../data.hpp"
#include "../mexutils.h"
#include <algorithm>
#include <limits>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------- */
/*                                                simpooling_*_forward */
/* ---------------------------------------------------------------- */

/*
 Symmetry-pooling implementation
 */

template<typename type> static inline void
simpooling_forward_cpu(type* pooled,
                    type const* data,
                    size_t rings, size_t angles, size_t depth)
{

	//We generate the half number of angles at output
	int pooledAngle = angles/2;
	type scale=type(1)/type(rings*pooledAngle);
	//We fuse all rings into one
	int pooledRing = 1;
	int idx_init[2*pooledAngle],idx[2*pooledAngle];

	//Matrix of indexes for simmetry
	for (int x = 0; x < pooledAngle; ++x) {
		idx_init[2*x]=x;
		idx_init[2*x+1]=angles-1-x;
	}
	//Channels
	for (int z = 0; z < depth; ++z) {
		//Re-init the idx vector
		memcpy(idx, idx_init, sizeof(int) * 2 * pooledAngle);
		//Angles loop
		for (int x = 0; x < pooledAngle; ++x) {
			pooled[x]=0;
			//Only the half of the angles
			for(int x1 = 0; x1 < pooledAngle; ++x1) {
				//For each ring we compute the differences and accumulate
				for (int y1 = 0; y1 < rings; ++y1) {
					//Compute and accumulate the difference among the sectors
					pooled[x]+=fabs(data[idx[2*x1]*rings + y1]-data[idx[2*x1+1]*rings + y1])*scale;
				}
			}
			//We need to perform the circular shift of the idx
			for (int x1 = 0; x1 < angles; ++x1) {
				idx[x1]=idx[x1]+1;
				if(idx[x1]==angles)
					idx[x1]=0;
			}
		}
		//Moving to the next channel
		data += angles*rings;
		pooled += pooledAngle;
	}
}

/* ---------------------------------------------------------------- */
/*                                               simpooling_*_backward */
/* ---------------------------------------------------------------- */

/*
 Symmetry pooling backward step
 */


template<typename type> static inline void
simpooling_backward_cpu(type* derData,
                     type const* data,
                     type const* derPooled,
                     size_t rings, size_t angles, size_t depth)
{
	//We generate the half number of angles at output
	int pooledAngle = angles/2;
	type scale=type(1.0)/type(2.0*rings*pooledAngle);
	int sign=1;
	//Este a 1 porque sumaremos todas las contribuciones
	int pooledRing = 1;
	int idx_init[2*pooledAngle],idx[2*pooledAngle];

	//Matrix of indexes for symmetry
	for (int x = 0; x < pooledAngle; ++x) {
		idx_init[2*x]=x;
		idx_init[2*x+1]=angles-1-x;
	}



	//Channels Loop
	for (int z = 0; z < depth; ++z) {
		//Re-init the idx vector
		memcpy(idx, idx_init, sizeof(int) * 2 * pooledAngle);
		//Angles loop
		for (int x = 0; x < pooledAngle; ++x) {
			//Only the half of the angles
			for(int x1 = 0; x1 < pooledAngle; ++x1) {
				//For each ring we compute the differences and accumulate
				for (int y1 = 0; y1 < rings; ++y1) {
					//Depending on the sign of the difference => We have to change the values of the mask
					//In forward we simply do "abs()" => here we need to know the sign to update things

					if((data[idx[2*x1]*rings + y1]-data[idx[2*x1+1]*rings + y1])>0)
						sign=1;
					else
						sign=-1;
					//Update the derivative of the z with respect to the data
					derData[idx[2*x1]*rings + y1]+=sign*derPooled[x]*scale;
					derData[idx[2*x1+1]*rings + y1]-=sign*derPooled[x]*scale;
				}
			}
			//We need to perform the circular shift of the idx
			for (int x1 = 0; x1 < angles; ++x1) {
				idx[x1]=idx[x1]+1;
				if(idx[x1]==angles)
					idx[x1]=0;
			}
		}
		//Moving to the next channel
		data += angles*rings;
		derData += angles*rings;
		derPooled += pooledAngle;
	}
}



/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct simpooling<vl::VLDT_CPU, type>
  {
    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            size_t rings, size_t angles, size_t depth)
    {
      simpooling_forward_cpu<type> (pooled,
    		  	  	  	  	  	    data,
    		  	  	  	  	  	    rings, angles, depth) ;
      return VLE_Success ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             type const* derOutput,
             size_t rings, size_t angles, size_t depth)
    {
      simpooling_backward_cpu<type> (derData,
                                     data, derOutput,
                                     rings, angles, depth) ;
      return VLE_Success ;
    }
 }; // simpooling

} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::simpooling<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::simpooling<vl::VLDT_CPU, double> ;
#endif

