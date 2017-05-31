// @file circ_pooling_cpu.cpp
// @brief Polar Pooling block implementation (cpu)
// @author Iván González Díaz

#include "circpooling_mask.hpp"
#include "../data.hpp"
#include <algorithm>
#include <limits>
#include <math.h>
#include "../mexutils.h"

/* ---------------------------------------------------------------- */
/*                                               Max pooling helper */
/* ---------------------------------------------------------------- */

template <typename type>
struct acc_max
{

  inline acc_max(int poolRings = 0, int poolAngles = 0, type derOutput = 0)
  :
  value(-std::numeric_limits<type>::infinity()),
  derOutput(derOutput),
  derDataActivePt(NULL)
  { }

  inline void init(int poolRings, int poolAngles, type aderOutput = 0){
    value=-std::numeric_limits<type>::infinity();
    derOutput = aderOutput;
    derDataActivePt = NULL;
  }


  inline void accumulate_forward(type x) {
    value = std::max(value, x) ;
  }

  inline void accumulate_scale(){
     }

  inline void accumulate_backward(type const* data, type* derDataPt) {
    type x = *data ;
    if (x > value) {
      value = x ;
      derDataActivePt = derDataPt ;
    }
  }

  inline type done_forward() const {
    return value ;
  }

  inline void done_backward() const {
    if (derDataActivePt) { *derDataActivePt += derOutput ; }
  }

  type value ;
  type derOutput ;
  type* derDataActivePt ;
} ;

/* ---------------------------------------------------------------- */
/*                                           Average pooling helper */
/* ---------------------------------------------------------------- */

template <typename type>
struct acc_sum
{
  inline acc_sum(int poolRings = 1, int poolAngles = 1, type derOutput = 0)
  :
  value(0),
  scale(0),//type(1)/type(poolRings*poolAngles)),
  derOutput(derOutput)
  { }

  inline void init(int poolRings, int poolAngles, type aderOutput = 0){
	  scale= 0;//type(1.0/(poolRings*poolAngles));
	  value = 0;
	  derOutput = aderOutput;
  }

  inline void accumulate_forward(type x) {
    value += x ;
    scale += type(1.0) ;
  }

  inline void accumulate_scale(){
  	  scale += type(1.0) ;
    }

  /* note: data is unused */
  inline void accumulate_backward(type const* data, type* derDataPt) {
	 //mexPrintf("scale %f\n",scale);mexEvalString("drawnow");
    *derDataPt += (derOutput / (scale + type(0.00001)));
  }

  inline type done_forward() const {
	  /*if(scale==type(0.0)){
		  mexPrintf("Warning!!!: Circpooling without data in a cell!!!\n");mexEvalString("drawnow");
		  return type(-2000);
	  }*/
	  return value / (scale + type(0.00001));
  }

  inline void done_backward() const { }

  type value ;
  type derOutput ;
  type scale ;
} ;

/* ---------------------------------------------------------------- */
/*											 circ_pooling_*_forward */
/* ---------------------------------------------------------------- */


template<typename type, typename Accumulator> static inline void
circpooling_mask_forward_cpu(type* pooled,
                    type const* data,
                    type const* pcoords,
                    size_t height, size_t width, size_t depth, size_t channels,
                    size_t windowRings, size_t windowAngles,
					float strideX, float strideY,
                    size_t padLeft, size_t padRight, size_t padTop, size_t padBottom)
{

  float xc,yc,minr,maxr,mina[2],maxa[2];
  int qa,qr,contChannels=0,contImages = 0;
  Accumulator accs[windowRings*windowAngles];
  int pcoordsDepth=2;
  int offsetAngle = width*height;

  //float *polarCoor=(float *)malloc(height*width*sizeof(float));
  //generatePolarCoordinates(height,width,polarCoor);
  float radius, angle;
  float stepAngle=2*PI/((float)windowAngles);
  type maxRadius = 0.0, minRadius=10.0;//Get the maximum radius
  for (int y = 0; y < height; ++y)
	  for (int x = 0; x < width; ++x){
		  maxRadius=std::max(maxRadius,pcoords[x * height + y]);
		  //minRadius=std::max(minRadius,pcoords[x * height + y]);
	  }
  float stepRadius = maxRadius/(float)(windowRings);

  //Bucle de channels e imágenes
  for (int z = 0; z < depth; ++z) {
	  contChannels++;
	  /*Initialization*/
	  for (int r = 0; r < windowRings; ++r) {
		  for (int a = 0; a < windowAngles; ++a) {
			  accs[a * windowRings + r].init(windowRings,windowAngles);
		  }
	  }

	  for (int y = 0; y < height; ++y) {
		  for (int x = 0; x < width; ++x) {
			  radius = pcoords[x * height + y];

			  //mexPrintf("%f\n",radius);mexEvalString("drawnow");
			  //If we are inside the lesion
			  if(radius>=0){
				  radius=radius-0.0001;
				  angle = pcoords[offsetAngle + x * height + y]-0.0001 ;
				  qr=(int)(radius/stepRadius);
				  //0-2*pi
				  qa = (int)(angle/stepAngle);
				  //mexPrintf("radius/angle %d/%d r %f sr %f a %f sa %f val %f\n",qr,qa,radius,stepRadius,angle,stepAngle,data[x * height + y]);mexEvalString("drawnow");
				  qr=std::min(qr,(int)(windowRings-1));
				  qa=std::min(qa,(int)(windowAngles-1));

				  accs[qa * windowRings + qr].accumulate_forward(data[x * height + y]);
			  }

		  }
		 // mexPrintf("\n");mexEvalString("drawnow");
	  }
	  //Finish accumulation
	  for (int r = 0; r < windowRings; ++r) {
		  for (int a = 0; a < windowAngles; ++a) {

			  pooled[a * windowRings + r] = accs[a * windowRings + r].done_forward() ;
		  }
		  //mexPrintf("\n");mexEvalString("drawnow");
	  }

	  //Advance a channel (or image)
	  data += width*height ;
	  pooled += windowAngles*windowRings ;
	  //Advance an image in pcoords
	  if(contChannels==channels){
		  pcoords += 2*width*height;
		  contChannels=0;
		  contImages+=1;
	  }
  }
}

/* ---------------------------------------------------------------- */
/*                                               pooling_*_backward */
/* ---------------------------------------------------------------- */

/*
 assume the output array to be cleared or otherwise
 properly initialised: accumulates the derivative
 */

/* Todo: transpose */

template<typename type, typename Accumulator> static inline void
circpooling_mask_backward_cpu(type* derData,
                     type const* data,
                     type const* pcoords,
                     type const* derPooled,
                     size_t height, size_t width, size_t depth, size_t channels,
                     size_t windowRings, size_t windowAngles,
					 float strideX, float strideY,
                     size_t padLeft, size_t padRight, size_t padTop, size_t padBottom)
{

	float xc,yc;
	int qa,qr;
	Accumulator accs[windowRings*windowAngles];
	int pcoordsDepth=2,contChannels=0;
	int offsetAngle = width*height;

	//float *polarCoor=(float *)malloc(height*width*sizeof(float));
	//generatePolarCoordinates(height,width,polarCoor);
	float radius, angle;
	float stepAngle=2*PI/((float)windowAngles);
	type maxRadius = 0.0;//Get the maximum radius
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
			maxRadius=std::max(maxRadius,pcoords[x * height + y]);
	float stepRadius = maxRadius/(float)(windowRings);


	for (int z = 0; z < depth; ++z) {
		contChannels++;
		/*Initialization*/
		for (int r = 0; r < windowRings; ++r) {
			for (int a = 0; a < windowAngles; ++a) {
				accs[a * windowRings + r].init(windowRings,windowAngles,derPooled[a * windowRings + r]);
			}
		}
		//Primero tenemos que acumular escala
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				radius = pcoords[x * height + y];
				//If we are inside the lesion
				if(radius>=0){
					radius=radius-0.0001;
					angle = pcoords[offsetAngle + x * height + y]-0.0001 ;
					qr=(int)(radius/stepRadius);
					//0-2*pi
					qa = (int)(angle/stepAngle);
					qr=std::min(qr,(int)(windowRings-1));
					qa=std::min(qa,(int)(windowAngles-1));
					accs[qa * windowRings + qr].accumulate_scale();
				}
			}
		}
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				radius = pcoords[x * height + y];
				//Inside the lesion
				if(radius>=0){
					radius=radius-0.0001;
					angle = pcoords[offsetAngle + x * height + y] -0.0001 ;
					qr=(int)(radius/stepRadius);
					qr=std::min(qr,(int)(windowRings-1));
					qa = (int)(angle/stepAngle);
					qa=std::min(qa,(int)(windowAngles-1));
					accs[qa * windowRings + qr].accumulate_backward(&data[x * height + y],&derData[x * height + y]);
				}
			}
		}
		//Finish accumulation
		for (int r = 0; r < windowRings; ++r) {
			for (int a = 0; a < windowAngles; ++a) {
				accs[a * windowRings + r].done_backward() ;
			}
		}
		data += width*height ;
		derData += width*height ;
		derPooled += windowAngles*windowRings ;
		//Advance an image in pcoords
		if(contChannels==channels){
			pcoords += 2*width*height;
			contChannels=0;
		}
	}
}

/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct circpooling_mask_max<vl::VLDT_CPU, type>
  {
    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            type const* pcoords,
            size_t height, size_t width, size_t depth, size_t channels,
            size_t poolRings, size_t poolAngles,
			float strideY, float strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      circpooling_mask_forward_cpu<type, acc_max<type> > (pooled,
                                                 data,
                                                 pcoords,
                                                 height, width, depth, channels,
                                                 poolRings, poolAngles,
                                                 strideY, strideX,
                                                 padTop, padBottom, padLeft, padRight) ;
      return VLE_Success ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             type const* pcoords,
             type const* derOutput,
             size_t height, size_t width, size_t depth, size_t channels,
             size_t poolRings, size_t poolAngles,
             float strideY, float strideX,
             size_t padTop, size_t padBottom,
             size_t padLeft, size_t padRight)
    {
      circpooling_mask_backward_cpu<type, acc_max<type> > (derData,
                                                  data, pcoords,derOutput,
                                                  height, width, depth, channels,
                                                  poolRings, poolAngles,
                                                  strideY, strideX,
                                                  padTop, padBottom, padLeft, padRight) ;
      return VLE_Success ;
    }
  } ; // pooling_max

  template <typename type>
  struct circpooling_mask_average<vl::VLDT_CPU, type>
  {

    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            type const* pcoords,
            size_t height, size_t width, size_t depth, size_t channels,
            size_t poolRings, size_t poolAngles,
			float strideY, float strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      circpooling_mask_forward_cpu<type, acc_sum<type> > (pooled,
                                                 data,
                                                 pcoords,
                                                 height, width, depth, channels,
                                                 poolRings, poolAngles,
                                                 strideY, strideX,
                                                 padTop, padBottom, padLeft, padRight) ;
      return VLE_Success ;
    }

    static vl::ErrorCode
    backward(type* derData,
    		 type const* pcoords,
             type const* derPooled,
             size_t height, size_t width, size_t depth, size_t channels,
             size_t poolRings, size_t poolAngles,
			 float strideY, float strideX,
             size_t padTop, size_t padBottom,
             size_t padLeft, size_t padRight)
    {
      circpooling_mask_backward_cpu<type, acc_sum<type> > (derData,
                                                  NULL, pcoords, derPooled,
                                                  height, width, depth, channels,
                                                  poolRings, poolAngles,
                                                  strideY, strideX,
                                                  padTop, padBottom, padLeft, padRight) ;
      return VLE_Success ;
    }
  } ; // pooling_average

} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::circpooling_mask_max<vl::VLDT_CPU, float> ;
template struct vl::impl::circpooling_mask_average<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::circpooling_mask_max<vl::VLDT_CPU, double> ;
template struct vl::impl::circpooling_mask_average<vl::VLDT_CPU, double> ;
#endif

