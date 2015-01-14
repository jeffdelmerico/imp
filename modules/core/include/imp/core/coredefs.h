/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : ImageUtilities
 * Module      : Core
 * Class       : IuSize, IuRect
 * Language    : C++
 * Description : Typedefinitions and Macros for ImageUtilities core module.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IU_COREDEFS_H
#define IU_COREDEFS_H

#include <cstdint>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include "globaldefs.h"

 /** Basic assert macro
 * This macro should be used to enforce any kind of pre or post conditions.
 * Unlike the C assertion this assert also prints an error/warning as output in release mode.
 * \note The macro is written in such a way that omitting a semicolon after its usage
 * causes a compiler error. The correct way to invoke this macro is:
 * IU_ASSERT(small_value < big_value);
 */
#ifdef DEBUG
#define IU_ASSERT(C) \
  do { \
    if (!(C)) \
    { \
      fprintf(stderr, "%s(%d) : assertion '%s' failed!\n", \
    __FILE__, __LINE__, #C ); \
    abort(); \
    } \
  } while(false)
#else //DEBUG
#define IU_ASSERT(C)
#endif //DEBUG


/** Error status codes.
 * Negative error codes represent an error.
 * Zero means that everything is ok.
 * Positive error codes represent warnings.
 */
typedef enum
{
  // error
  IU_MEM_COPY_ERROR = -11,
  IU_MEM_ALLOC_ERROR = -10,
  IU_CUDA_ERROR = -3,
  IU_NOT_SUPPORTED_ERROR = -2,
  IU_ERROR = -1,

  // success
  IU_NO_ERROR = 0,
  IU_SUCCESS = 0,

  // warnings
  IU_WARNING = 1

} IuStatus;



typedef enum
{
  IU_EQUAL,
  IU_NOT_EQUAL,
  IU_GREATER,
  IU_GREATER_EQUAL,
  IU_LESS,
  IU_LESS_EQUAL
} IuComparisonOperator;

typedef enum
{
  IU_NO  = 0, // no function
  IU_ABS = 1, // abs(x)
  IU_SQR = 2, // x*x
  IU_CNT = 3  // 1 if neq 0
} IuSparseSum;

typedef enum
{
  COO = 0, // uncompressed sparse format
  CSR = 1, // compressed rows
  CSC = 2  // compressed columns
} IuSparseFormat;

/** Interpolation types. */
typedef enum
{
  IU_INTERPOLATE_NEAREST, /**< nearest neighbour interpolation. */
  IU_INTERPOLATE_LINEAR, /**< linear interpolation. */
  IU_INTERPOLATE_CUBIC, /**< cubic interpolation. */
  IU_INTERPOLATE_CUBIC_SPLINE /**< cubic spline interpolation. */
} IuInterpolationType;



inline bool operator==(const IuRect& a, const IuRect& b)
{
  return ((a.x == b.x) && (a.y == b.y) &&
          (a.width == b.width) && (a.height == b.height));
}

inline bool operator!=(const IuRect& a, const IuRect& b)
{
  return ((a.x != b.x) || (a.y != b.y) ||
          (a.width != b.width) || (a.height != b.height));
}

/** 3D Cube
 * This struct contains cordinates of upper left corner and its size in pixels.
 */
struct IuCube
{
  int x;       //!< x-coord of the upper left corner
  int y;       //!< y-coord of the upper left corner
  int z;       //!< z-coord of the upper left corner
  unsigned int width;   //!< width of the rectangle
  unsigned int height;  //!< height of the rectangle
  unsigned int depth;  //!< depth of the rectangle

  IuCube() :
      x(0), y(0), z(0), width(0), height(0), depth(0)
  {
  }

  IuCube(int _x, int _y, int _z, unsigned int _width, unsigned int _height, unsigned int _depth) :
      x(_x), y(_y), z(_z), width(_width), height(_height), depth(_depth)
  {
  }

  IuCube(const IuCube& from) :
      x(from.x), y(from.y), z(from.z), width(from.width), height(from.height), depth(from.depth)
  {
  }

  IuCube& operator= (const IuCube& from)
  {
//    if (from == *this)
//      return *this;

    this->x = from.x;
    this->y = from.y;
    this->z = from.z;
    this->width = from.width;
    this->height = from.height;
    this->depth = from.depth;

    return *this;
  }

  IuCube(const IuSize& from) :
      x(0), y(0), z(0), width(from.width), height(from.height), depth(from.depth)
  {
  }

  IuCube& operator= (const IuSize& from)
  {
    this->x = 0;
    this->y = 0;
    this->z = 0;
    this->width = from.width;
    this->height = from.height;
    this->depth = from.depth;

    return *this;
  }

};

#endif // IU_COREDEFS_H
