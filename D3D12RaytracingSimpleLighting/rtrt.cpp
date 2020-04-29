#pragma once
#include "stdafx.h"
#include "rtrt.h"

namespace rtrt
{
  AccelerationStructure::AccelerationStructure() :
    scratch(nullptr),
    instance_descs_buffer(nullptr)
  {

  }

  AccelerationStructure::~AccelerationStructure()
  {
    for (size_t i = 0; i < structures.size(); i++)
    {
      RELEASE(structures[i]);
    }

    if (instance_descs_buffer != nullptr)
    {
      delete instance_descs_buffer;
    }

    RELEASE(scratch);
  }

  
 

}