#pragma once
namespace rtrt
{
	struct AccelerationStructure
	{
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE type;
		ID3D12Resource* scratch = nullptr;
		std::vector<ID3D12Resource*> structures;
		std::vector<WRAPPED_GPU_POINTER> structure_pointers;
		Buffer* instance_descs_buffer = nullptr;//tlas

		AccelerationStructure();
		~AccelerationStructure();
	};

}