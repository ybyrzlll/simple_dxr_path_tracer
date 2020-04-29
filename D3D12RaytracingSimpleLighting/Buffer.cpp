#pragma once
#include "stdafx.h"
#include "Buffer.h"

namespace rtrt{
		Buffer::Buffer() :
			buffer_(nullptr),
			size_(0)
		{

		}

		Buffer::~Buffer()
		{
			RELEASE_SAFE(buffer_);
		}

		ID3D12Resource * Buffer::GetBuffer()
		{
			return buffer_;
		}
}
