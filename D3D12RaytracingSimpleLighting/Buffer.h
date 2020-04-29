#pragma once
namespace rtrt {
	class Buffer
	{
	public:
		Buffer();
		~Buffer();

		ID3D12Resource* GetBuffer();

		ID3D12Resource* buffer_;
		UINT size_;
	};
}


