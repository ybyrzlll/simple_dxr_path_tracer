//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include "stdafx.h"
#include "D3D12RaytracingSimpleLighting.h"
#include "DirectXRaytracingHelper.h"
#include "CompiledShaders\Raytracing.hlsl.h"

using namespace DX;

//const wchar_t* D3D12RaytracingSimpleLighting::c_hitGroupName = L"MyHitGroup";
const wchar_t* D3D12RaytracingSimpleLighting::c_raygenShaderName = L"MyRaygenShader";
const wchar_t* D3D12RaytracingSimpleLighting::c_closestHitShaderName = L"MyClosestHitShader";
const wchar_t* D3D12RaytracingSimpleLighting::c_missShaderNames[] =
{
	L"MyMissShader", L"MyMissShader_ShadowRay"
};

// Hit groups.
const wchar_t* D3D12RaytracingSimpleLighting::c_hitGroupNames_TriangleGeometry[] =
{
	L"MyHitGroup_Triangle", L"MyHitGroup_Triangle_ShadowRay"
};

D3D12RaytracingSimpleLighting::D3D12RaytracingSimpleLighting(UINT width, UINT height, std::wstring name) :
	DXSample(width, height, name),
	m_raytracingOutputResourceUAVDescriptorHeapIndex(UINT_MAX),
	m_curRotationAngleRad(0.0f),
	m_isDxrSupported(false)
{
	m_forceComputeFallback = false;
	SelectRaytracingAPI(RaytracingAPI::FallbackLayer);
	UpdateForSizeChange(width, height);
}

void D3D12RaytracingSimpleLighting::EnableDirectXRaytracing(IDXGIAdapter1* adapter)
{
	// Fallback Layer uses an experimental feature and needs to be enabled before creating a D3D12 device.
	bool isFallbackSupported = EnableComputeRaytracingFallback(adapter);

	if (!isFallbackSupported)
	{
		OutputDebugString(
			L"Warning: Could not enable Compute Raytracing Fallback (D3D12EnableExperimentalFeatures() failed).\n" \
			L"         Possible reasons: your OS is not in developer mode.\n\n");
	}

	m_isDxrSupported = IsDirectXRaytracingSupported(adapter);

	if (!m_isDxrSupported)
	{
		OutputDebugString(L"Warning: DirectX Raytracing is not supported by your GPU and driver.\n\n");

		ThrowIfFalse(isFallbackSupported,
			L"Could not enable compute based fallback raytracing support (D3D12EnableExperimentalFeatures() failed).\n"\
			L"Possible reasons: your OS is not in developer mode.\n\n");
		m_raytracingAPI = RaytracingAPI::FallbackLayer;
	}
}

void D3D12RaytracingSimpleLighting::OnInit()
{
	m_deviceResources = std::make_unique<DeviceResources>(
		DXGI_FORMAT_R8G8B8A8_UNORM,
		DXGI_FORMAT_UNKNOWN,
		FrameCount,
		D3D_FEATURE_LEVEL_11_0,
		// Sample shows handling of use cases with tearing support, which is OS dependent and has been supported since TH2.
		// Since the Fallback Layer requires Fall Creator's update (RS3), we don't need to handle non-tearing cases.
		DeviceResources::c_RequireTearingSupport,
		m_adapterIDoverride
		);
	m_deviceResources->RegisterDeviceNotify(this);
	m_deviceResources->SetWindow(Win32Application::GetHwnd(), m_width, m_height);
	m_deviceResources->InitializeDXGIAdapter();
	EnableDirectXRaytracing(m_deviceResources->GetAdapter());

	m_deviceResources->CreateDeviceResources();
	m_deviceResources->CreateWindowSizeDependentResources();

	InitializeScene();

	CreateDeviceDependentResources();
	CreateWindowSizeDependentResources();
}

// Update camera matrices passed into the shader.
void D3D12RaytracingSimpleLighting::UpdateCameraMatrices()
{
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

	m_sceneCB[frameIndex].cameraPosition = m_eye;
	float fovAngleY = 45.0f;
	XMMATRIX view = XMMatrixLookAtLH(m_eye, m_at, m_up);
	XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(fovAngleY), m_aspectRatio, 1.0f, 125.0f);
	XMMATRIX viewProj = view * proj;

	m_sceneCB[frameIndex].projectionToWorld = XMMatrixInverse(nullptr, viewProj);
}

// Initialize scene rendering parameters.
void D3D12RaytracingSimpleLighting::InitializeScene()
{
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

	// Setup materials.
	{
		//m_cubeCB.albedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
		m_cubeCB = { XMFLOAT4(0.9f, 0.9f, 0.9f, 1.0f), 0.25f, 1, 0.4f, 50, 1 };
	}

	// Setup camera.
	{
		// Initialize the view and projection inverse matrices.
		m_eye = { 0.0f, 10.0f, -5.0f, 1.0f };
		m_at = { 0.0f, 0.0f, 0.0f, 1.0f };
		m_up = { 0.0f, 1.0f, 0.0f, 1.0f };

		// Rotate camera around Y axis.
	   /* XMMATRIX rotate = XMMatrixRotationY(XMConvertToRadians(45.0f));
		m_eye = XMVector3Transform(m_eye, rotate);
		m_up = XMVector3Transform(m_up, rotate);*/

		UpdateCameraMatrices();
	}

	// Setup lights.
	{
		// Initialize the lighting parameters.
		XMFLOAT4 lightPosition;
		XMFLOAT4 lightAmbientColor;
		XMFLOAT4 lightDiffuseColor;

		lightPosition = XMFLOAT4(0.0f, 5.8f, -2.0f, 0.0f);
		m_sceneCB[frameIndex].lightPosition = XMLoadFloat4(&lightPosition);

		lightAmbientColor = XMFLOAT4(0.5f, 0.5f, 0.5f, 1.0f);
		m_sceneCB[frameIndex].lightAmbientColor = XMLoadFloat4(&lightAmbientColor);

		lightDiffuseColor = XMFLOAT4(0.5f, 0.0f, 0.0f, 1.0f);
		m_sceneCB[frameIndex].lightDiffuseColor = XMLoadFloat4(&lightDiffuseColor);
	}

	// Apply the initial values to all frames' buffer instances.
	for (auto& sceneCB : m_sceneCB)
	{
		sceneCB = m_sceneCB[frameIndex];
	}
}

// Create constant buffers.
void D3D12RaytracingSimpleLighting::CreateConstantBuffers()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto frameCount = m_deviceResources->GetBackBufferCount();

	// Create the constant buffer memory and map the CPU and GPU addresses
	const D3D12_HEAP_PROPERTIES uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

	// Allocate one constant buffer per frame, since it gets updated every frame.
	size_t cbSize = frameCount * sizeof(AlignedSceneConstantBuffer);
	const D3D12_RESOURCE_DESC constantBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(cbSize);

	ThrowIfFailed(device->CreateCommittedResource(
		&uploadHeapProperties,
		D3D12_HEAP_FLAG_NONE,
		&constantBufferDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&m_perFrameConstants)));

	// Map the constant buffer and cache its heap pointers.
	// We don't unmap this until the app closes. Keeping buffer mapped for the lifetime of the resource is okay.
	CD3DX12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
	ThrowIfFailed(m_perFrameConstants->Map(0, nullptr, reinterpret_cast<void**>(&m_mappedConstantData)));
}

// Create resources that depend on the device.
void D3D12RaytracingSimpleLighting::CreateDeviceDependentResources()
{
	// Initialize raytracing pipeline.

	// Create raytracing interfaces: raytracing device and commandlist.
	CreateRaytracingInterfaces();

	// Create root signatures for the shaders.
	CreateRootSignatures();

	// Create a raytracing pipeline state object which defines the binding of shaders, state and resources to be used during raytracing.
	CreateRaytracingPipelineStateObject();

	// Create a heap for descriptors.
	CreateDescriptorHeap();

	// Build geometry to be used in the sample.
	BuildGeometry();

	// Build raytracing acceleration structures from the generated geometry.
	BuildAccelerationStructures();

	// Create constant buffers for the geometry and the scene.
	CreateConstantBuffers();

	// Build shader tables, which define shaders and their local root arguments.
	BuildShaderTables();

	// Create an output 2D texture to store the raytracing result to.
	CreateRaytracingOutputResource();
}

void D3D12RaytracingSimpleLighting::SerializeAndCreateRaytracingRootSignature(D3D12_ROOT_SIGNATURE_DESC& desc, ComPtr<ID3D12RootSignature>* rootSig)
{
	auto device = m_deviceResources->GetD3DDevice();
	ComPtr<ID3DBlob> blob;
	ComPtr<ID3DBlob> error;

	if (m_raytracingAPI == RaytracingAPI::FallbackLayer)
	{
		ThrowIfFailed(m_fallbackDevice->D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error), error ? static_cast<wchar_t*>(error->GetBufferPointer()) : nullptr);
		ThrowIfFailed(m_fallbackDevice->CreateRootSignature(1, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&(*rootSig))));
	}
	else // DirectX Raytracing
	{
		ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error), error ? static_cast<wchar_t*>(error->GetBufferPointer()) : nullptr);
		ThrowIfFailed(device->CreateRootSignature(1, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&(*rootSig))));
	}
}

void D3D12RaytracingSimpleLighting::CreateRootSignatures()
{
	auto device = m_deviceResources->GetD3DDevice();

	// Global Root Signature
	// This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
	{
		CD3DX12_DESCRIPTOR_RANGE ranges[2]; // Perfomance TIP: Order from most frequent to least frequent.
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture
		ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1);  // 2 static index and vertex buffers.

		CD3DX12_ROOT_PARAMETER rootParameters[GlobalRootSignatureParams::Count];
		rootParameters[GlobalRootSignatureParams::OutputViewSlot].InitAsDescriptorTable(1, &ranges[0]);
		rootParameters[GlobalRootSignatureParams::AccelerationStructureSlot].InitAsShaderResourceView(0);
		rootParameters[GlobalRootSignatureParams::SceneConstantSlot].InitAsConstantBufferView(0);
		rootParameters[GlobalRootSignatureParams::Meshes].InitAsShaderResourceView(CPP_REGISTER_MESHES);
		rootParameters[GlobalRootSignatureParams::Vertices].InitAsShaderResourceView(CPP_REGISTER_VERTICES);
		rootParameters[GlobalRootSignatureParams::Indices].InitAsShaderResourceView(CPP_REGISTER_INDICES);

		CD3DX12_ROOT_SIGNATURE_DESC globalRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
		SerializeAndCreateRaytracingRootSignature(globalRootSignatureDesc, &m_raytracingGlobalRootSignature);
	}

	// Local Root Signature
	// This is a root signature that enables a shader to have unique arguments that come from shader tables.
	{
		CD3DX12_ROOT_PARAMETER rootParameters[LocalRootSignatureParams::Count];
		rootParameters[LocalRootSignatureParams::CubeConstantSlot].InitAsConstants(SizeOfInUint32(m_cubeCB), 1);
		CD3DX12_ROOT_SIGNATURE_DESC localRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
		localRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
		SerializeAndCreateRaytracingRootSignature(localRootSignatureDesc, &m_raytracingLocalRootSignature);
	}
}

// Create raytracing device and command list.
void D3D12RaytracingSimpleLighting::CreateRaytracingInterfaces()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto commandList = m_deviceResources->GetCommandList();

	if (m_raytracingAPI == RaytracingAPI::FallbackLayer)
	{
		CreateRaytracingFallbackDeviceFlags createDeviceFlags = m_forceComputeFallback ?
			CreateRaytracingFallbackDeviceFlags::ForceComputeFallback :
			CreateRaytracingFallbackDeviceFlags::None;
		ThrowIfFailed(D3D12CreateRaytracingFallbackDevice(device, createDeviceFlags, 0, IID_PPV_ARGS(&m_fallbackDevice)));
		m_fallbackDevice->QueryRaytracingCommandList(commandList, IID_PPV_ARGS(&m_fallbackCommandList));
	}
	else // DirectX Raytracing
	{
		ThrowIfFailed(device->QueryInterface(IID_PPV_ARGS(&m_dxrDevice)), L"Couldn't get DirectX Raytracing interface for the device.\n");
		ThrowIfFailed(commandList->QueryInterface(IID_PPV_ARGS(&m_dxrCommandList)), L"Couldn't get DirectX Raytracing interface for the command list.\n");
	}
}

// Local root signature and shader association
// This is a root signature that enables a shader to have unique arguments that come from shader tables.
void D3D12RaytracingSimpleLighting::CreateLocalRootSignatureSubobjects(CD3D12_STATE_OBJECT_DESC* raytracingPipeline)
{
	// Ray gen and miss shaders in this sample are not using a local root signature and thus one is not associated with them.

	// Local root signature to be used in a hit group.
	auto localRootSignature = raytracingPipeline->CreateSubobject<CD3D12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
	localRootSignature->SetRootSignature(m_raytracingLocalRootSignature.Get());
	// Define explicit shader association for the local root signature. 
	{
		auto rootSignatureAssociation = raytracingPipeline->CreateSubobject<CD3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
		rootSignatureAssociation->SetSubobjectToAssociate(*localRootSignature);
		rootSignatureAssociation->AddExports(c_hitGroupNames_TriangleGeometry);
	}
}

// Create a raytracing pipeline state object (RTPSO).
// An RTPSO represents a full set of shaders reachable by a DispatchRays() call,
// with all configuration options resolved, such as local signatures and other state.
void D3D12RaytracingSimpleLighting::CreateRaytracingPipelineStateObject()
{
	// Create 7 subobjects that combine into a RTPSO:
	// Subobjects need to be associated with DXIL exports (i.e. shaders) either by way of default or explicit associations.
	// Default association applies to every exported shader entrypoint that doesn't have any of the same type of subobject associated with it.
	// This simple sample utilizes default shader association except for local root signature subobject
	// which has an explicit association specified purely for demonstration purposes.
	// 1 - DXIL library
	// 1 - Triangle hit group
	// 1 - Shader config
	// 2 - Local root signature and association
	// 1 - Global root signature
	// 1 - Pipeline config
	CD3D12_STATE_OBJECT_DESC raytracingPipeline{ D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE };


	// DXIL library
	// This contains the shaders and their entrypoints for the state object.
	// Since shaders are not considered a subobject, they need to be passed in via DXIL library subobjects.
	auto lib = raytracingPipeline.CreateSubobject<CD3D12_DXIL_LIBRARY_SUBOBJECT>();
	D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE((void *)g_pRaytracing, ARRAYSIZE(g_pRaytracing));
	lib->SetDXILLibrary(&libdxil);
	// Define which shader exports to surface from the library.
	// If no shader exports are defined for a DXIL library subobject, all shaders will be surfaced.
	// In this sample, this could be ommited for convenience since the sample uses all shaders in the library. 
	{
		/*lib->DefineExport(c_raygenShaderName);
		lib->DefineExport(c_closestHitShaderName);
		lib->DefineExport(c_missShaderName);*/
	}

	// Triangle hit group
	// A hit group specifies closest hit, any hit and intersection shaders to be executed when a ray intersects the geometry's triangle/AABB.
	// In this sample, we only use triangle geometry with a closest hit shader, so others are not set.
	{
		auto hitGroup = raytracingPipeline.CreateSubobject<CD3D12_HIT_GROUP_SUBOBJECT>();
		hitGroup->SetClosestHitShaderImport(c_closestHitShaderName);
		hitGroup->SetHitGroupExport(c_hitGroupNames_TriangleGeometry[RayType::Radiance]);
		hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);
	}
	
	{
		auto hitGroup = raytracingPipeline.CreateSubobject<CD3D12_HIT_GROUP_SUBOBJECT>();
		hitGroup->SetHitGroupExport(c_hitGroupNames_TriangleGeometry[RayType::Shadow]);
		hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);
	}
	
	// Shader config
	// Defines the maximum sizes in bytes for the ray payload and attribute structure.
	auto shaderConfig = raytracingPipeline.CreateSubobject<CD3D12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
	/*RayPayload test;
	XMFLOAT4 a;
	UINT t1 = sizeof(test.color), t2 = sizeof(test.recursionDepth), t3 = sizeof(a);*/
	UINT payloadSize = max(sizeof(RayPayload), sizeof(ShadowRayPayload));
	// The maximum number of scalars (counted as 4 bytes each) that can be used for attributes in pipelines that contain this shader. The value cannot exceed D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES.
	UINT attributeSize = 2 * 4;
	shaderConfig->Config(payloadSize, attributeSize);

	// Local root signature and shader association
	// This is a root signature that enables a shader to have unique arguments that come from shader tables.
	CreateLocalRootSignatureSubobjects(&raytracingPipeline);

	// Global root signature
	// This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
	auto globalRootSignature = raytracingPipeline.CreateSubobject<CD3D12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
	globalRootSignature->SetRootSignature(m_raytracingGlobalRootSignature.Get());

	// Pipeline config
	// Defines the maximum TraceRay() recursion depth.
	auto pipelineConfig = raytracingPipeline.CreateSubobject<CD3D12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
	// PERFOMANCE TIP: Set max recursion depth as low as needed 
	// as drivers may apply optimization strategies for low recursion depths.
	UINT maxRecursionDepth = 4;// MAX_RAY_RECURSION_DEPTH; // ~ primary rays only. 
	pipelineConfig->Config(maxRecursionDepth);

#if _DEBUG
	//PrintStateObjectDesc(raytracingPipeline);
#endif

	// Create the state object.
	if (m_raytracingAPI == RaytracingAPI::FallbackLayer)
	{
		ThrowIfFailed(m_fallbackDevice->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&m_fallbackStateObject)), L"Couldn't create DirectX Raytracing state object.\n");
	}
	else // DirectX Raytracing
	{
		ThrowIfFailed(m_dxrDevice->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&m_dxrStateObject)), L"Couldn't create DirectX Raytracing state object.\n");
	}
}

// Create 2D output texture for raytracing.
void D3D12RaytracingSimpleLighting::CreateRaytracingOutputResource()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto backbufferFormat = m_deviceResources->GetBackBufferFormat();

	// Create the output resource. The dimensions and format should match the swap-chain.
	auto uavDesc = CD3DX12_RESOURCE_DESC::Tex2D(backbufferFormat, m_width, m_height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	ThrowIfFailed(device->CreateCommittedResource(
		&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &uavDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_raytracingOutput)));
	NAME_D3D12_OBJECT(m_raytracingOutput);

	D3D12_CPU_DESCRIPTOR_HANDLE uavDescriptorHandle;
	m_raytracingOutputResourceUAVDescriptorHeapIndex = AllocateDescriptor(&uavDescriptorHandle, m_raytracingOutputResourceUAVDescriptorHeapIndex);
	D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
	UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	device->CreateUnorderedAccessView(m_raytracingOutput.Get(), nullptr, &UAVDesc, uavDescriptorHandle);
	m_raytracingOutputResourceUAVGpuDescriptor = CD3DX12_GPU_DESCRIPTOR_HANDLE(m_descriptorHeap->GetGPUDescriptorHandleForHeapStart(), m_raytracingOutputResourceUAVDescriptorHeapIndex, m_descriptorSize);
}

void D3D12RaytracingSimpleLighting::CreateDescriptorHeap()
{
	auto device = m_deviceResources->GetD3DDevice();

	D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
	// Allocate a heap for 5 descriptors:
	// 6 - vertex and index buffer SRVs
	// 1 - raytracing output texture SRV
	// 4 - bottom and top level acceleration structure fallback wrapped pointer UAVs
	descriptorHeapDesc.NumDescriptors = 11; //!!!!!!!
	descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	descriptorHeapDesc.NodeMask = 0;
	device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&m_descriptorHeap));
	NAME_D3D12_OBJECT(m_descriptorHeap);

	m_descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

// Build geometry used in the sample.
void D3D12RaytracingSimpleLighting::BuildGeometry()
{
	m_indexBuffer.resize(ModelCount);
	m_vertexBuffer.resize(ModelCount);

	//������ΪAC�Ķ�������buffer
	BuildPlane();
	BuildCube();
	//BuildSphere();

	//������Ϊshaderʹ�õĶ�������buffer

	auto device = m_deviceResources->GetD3DDevice();

	meshes_buffer = new rtrt::Buffer();
	all_vertices_buffer = new rtrt::Buffer();
	all_indices_buffer = new rtrt::Buffer();

	meshes.resize(ModelCount);
	std::vector<Vertex> all_vertices;
	std::vector<Index> all_indices;

	for (int i = 0; i < ModelCount; i++)
	{
		meshes[i].first_idx_vertices = static_cast<UINT>(all_vertices.size());
		meshes[i].first_idx_indices = static_cast<UINT>(all_indices.size());
		//meshes[i].material = app.model.meshes[i].material;

		all_vertices.insert(all_vertices.end(), model.vertices[i].begin(), model.vertices[i].end());
		all_indices.insert(all_indices.end(), model.indices[i].begin(), model.indices[i].end());
	}

	AllocateUploadBuffer(device, meshes.data(), static_cast<UINT>(meshes.size() * sizeof(Mesh)), &meshes_buffer->buffer_);
	AllocateUploadBuffer(device, all_vertices.data(), static_cast<UINT>(all_vertices.size() * sizeof(Vertex)), &all_vertices_buffer->buffer_);
	AllocateUploadBuffer(device, all_indices.data(), static_cast<UINT>(all_indices.size() * sizeof(Index)), &all_indices_buffer->buffer_);

}

void D3D12RaytracingSimpleLighting::BuildPlane()
{
	auto device = m_deviceResources->GetD3DDevice();

	TCHAR pszAppPath[MAX_PATH] = {};
	// �õ���ǰ�Ĺ���Ŀ¼����������ʹ�����·�������ʸ�����Դ�ļ�
	{
		UINT nBytes = GetCurrentDirectory(MAX_PATH, pszAppPath);
		if (MAX_PATH == nBytes)
		{
			ThrowIfFalse(HRESULT_FROM_WIN32(GetLastError()));
		}
	}
	USES_CONVERSION;
	CHAR pszMeshFileName[MAX_PATH] = {};
	StringCchPrintfA(pszMeshFileName, MAX_PATH, "%s\\Mesh\\plane.obj", T2A(pszAppPath));

	UINT	nVertexCnt, nIndexCnt = 0;
	vector<Vertex> vertices;
	vector<Index> indices;

	LoadMeshVertex(pszMeshFileName, nVertexCnt, vertices, indices);

	m_indexBuffer[ModelType::Plane].count = m_vertexBuffer[ModelType::Plane].count = nIndexCnt = nVertexCnt;

	model.vertices.push_back(vertices);
	model.indices.push_back(indices);


	AllocateUploadBuffer(device, indices.data(), nIndexCnt * sizeof(Index), &m_indexBuffer[ModelType::Plane].resource);
	AllocateUploadBuffer(device, vertices.data(), nVertexCnt * sizeof(Vertex), &m_vertexBuffer[ModelType::Plane].resource);

	// Vertex buffer is passed to the shader along with index buffer as a descriptor table.
	// Vertex buffer descriptor must follow index buffer descriptor in the descriptor heap.
	UINT descriptorIndexIB = CreateBufferSRV(&m_indexBuffer[ModelType::Plane], (nIndexCnt * sizeof(Index)) / 4, 0);
	UINT descriptorIndexVB = CreateBufferSRV(&m_vertexBuffer[ModelType::Plane], nVertexCnt, sizeof(vertices[0]));
	ThrowIfFalse(descriptorIndexVB == descriptorIndexIB + 1, L"Vertex Buffer descriptor index must follow that of Index Buffer descriptor index!");
}

void D3D12RaytracingSimpleLighting::BuildSphere()
{
	auto device = m_deviceResources->GetD3DDevice();

	TCHAR pszAppPath[MAX_PATH] = {};
	// �õ���ǰ�Ĺ���Ŀ¼����������ʹ�����·�������ʸ�����Դ�ļ�
	{
		UINT nBytes = GetCurrentDirectory(MAX_PATH, pszAppPath);
		if (MAX_PATH == nBytes)
		{
			ThrowIfFalse(HRESULT_FROM_WIN32(GetLastError()));
		}
	}
	USES_CONVERSION;
	CHAR pszMeshFileName[MAX_PATH] = {};
	StringCchPrintfA(pszMeshFileName, MAX_PATH, "%s\\Mesh\\cube.txt", T2A(pszAppPath));

	UINT	nVertexCnt, nIndexCnt = 0;
	vector<Vertex> vertices;
	vector<Index> indices;

	LoadMeshVertex(pszMeshFileName, nVertexCnt, vertices, indices);
	m_indexBuffer[ModelType::Sphere].count = m_vertexBuffer[ModelType::Sphere].count = nIndexCnt = nVertexCnt;

	model.vertices.push_back(vertices);
	model.indices.push_back(indices);

	AllocateUploadBuffer(device, indices.data(), nIndexCnt * sizeof(Index), &m_indexBuffer[ModelType::Sphere].resource);
	AllocateUploadBuffer(device, vertices.data(), nVertexCnt * sizeof(Vertex), &m_vertexBuffer[ModelType::Sphere].resource);

	// Vertex buffer is passed to the shader along with index buffer as a descriptor table.
	// Vertex buffer descriptor must follow index buffer descriptor in the descriptor heap.
	UINT descriptorIndexIB = CreateBufferSRV(&m_indexBuffer[ModelType::Sphere], (nIndexCnt * sizeof(Index)) / 4, 0);
	UINT descriptorIndexVB = CreateBufferSRV(&m_vertexBuffer[ModelType::Sphere], nVertexCnt, sizeof(vertices[0]));
	ThrowIfFalse(descriptorIndexVB == descriptorIndexIB + 1, L"Vertex Buffer descriptor index must follow that of Index Buffer descriptor index!");
}

void D3D12RaytracingSimpleLighting::BuildCube()
{
	auto device = m_deviceResources->GetD3DDevice();

	TCHAR pszAppPath[MAX_PATH] = {};
	// �õ���ǰ�Ĺ���Ŀ¼����������ʹ�����·�������ʸ�����Դ�ļ�
	{
		UINT nBytes = GetCurrentDirectory(MAX_PATH, pszAppPath);
		if (MAX_PATH == nBytes)
		{
			ThrowIfFalse(HRESULT_FROM_WIN32(GetLastError()));
		}
	}
	USES_CONVERSION;
	CHAR pszMeshFileName[MAX_PATH] = {};
	StringCchPrintfA(pszMeshFileName, MAX_PATH, "%s\\Mesh\\sphere32.obj", T2A(pszAppPath));

	UINT	nVertexCnt, nIndexCnt = 0;
	vector<Vertex> vertices;
	vector<Index> indices;

	LoadMeshVertex(pszMeshFileName, nVertexCnt, vertices, indices);

	m_indexBuffer[ModelType::Cube].count = m_vertexBuffer[ModelType::Cube].count = nIndexCnt = nVertexCnt;

	model.vertices.push_back(vertices);
	model.indices.push_back(indices);


	AllocateUploadBuffer(device, indices.data(), nIndexCnt * sizeof(Index), &m_indexBuffer[ModelType::Cube].resource);
	AllocateUploadBuffer(device, vertices.data(), nVertexCnt * sizeof(Vertex), &m_vertexBuffer[ModelType::Cube].resource);

	// Vertex buffer is passed to the shader along with index buffer as a descriptor table.
	// Vertex buffer descriptor must follow index buffer descriptor in the descriptor heap.
	UINT descriptorIndexIB = CreateBufferSRV(&m_indexBuffer[ModelType::Cube], (nIndexCnt * sizeof(Index)) / 4, 0);
	UINT descriptorIndexVB = CreateBufferSRV(&m_vertexBuffer[ModelType::Cube], nVertexCnt, sizeof(vertices[0]));
	ThrowIfFalse(descriptorIndexVB == descriptorIndexIB + 1, L"Vertex Buffer descriptor index must follow that of Index Buffer descriptor index!");
}

// Build acceleration structures needed for raytracing.
void D3D12RaytracingSimpleLighting::BuildAccelerationStructures()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto commandList = m_deviceResources->GetCommandList();
	auto commandQueue = m_deviceResources->GetCommandQueue();
	auto commandAllocator = m_deviceResources->GetCommandAllocator();

	// Reset the command list for the acceleration structure construction.
	commandList->Reset(commandAllocator, nullptr);

	UINT num_model = m_indexBuffer.size();
	std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometryDesc;
	std::vector<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC> build_descs;
	std::vector<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO> prebuild_infos;

	m_bottomLevelAccelerationStructure.structures.resize(num_model);
	m_bottomLevelAccelerationStructure.structure_pointers.resize(num_model);
	geometryDesc.resize(num_model);
	build_descs.resize(num_model);
	prebuild_infos.resize(num_model);
	UINT64 max_size_scratch = 0;

	for (size_t i = 0; i < num_model; i++) {
		geometryDesc[i].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
		geometryDesc[i].Triangles.IndexBuffer = m_indexBuffer[i].resource->GetGPUVirtualAddress();
		geometryDesc[i].Triangles.IndexCount = static_cast<UINT>(m_indexBuffer[i].resource->GetDesc().Width) / sizeof(Index);
		geometryDesc[i].Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;//!!!!!!!!!!!!!
		geometryDesc[i].Triangles.Transform3x4 = 0;
		geometryDesc[i].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
		geometryDesc[i].Triangles.VertexCount = static_cast<UINT>(m_vertexBuffer[i].resource->GetDesc().Width) / sizeof(Vertex);
		geometryDesc[i].Triangles.VertexBuffer.StartAddress = m_vertexBuffer[i].resource->GetGPUVirtualAddress();
		geometryDesc[i].Triangles.VertexBuffer.StrideInBytes = sizeof(Vertex);
		// Mark the geometry as opaque. 
		// PERFORMANCE TIP: mark geometry as opaque whenever applicable as it can enable important ray processing optimizations.
		// Note: When rays encounter opaque geometry an any hit shader will not be executed whether it is present or not.
		geometryDesc[i].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

		build_descs[i] = {};
		build_descs[i].Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
		build_descs[i].Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
		build_descs[i].Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
		build_descs[i].Inputs.NumDescs = 1;
		build_descs[i].Inputs.pGeometryDescs = &geometryDesc[i];


		m_fallbackDevice->GetRaytracingAccelerationStructurePrebuildInfo(&build_descs[i].Inputs, &prebuild_infos[i]);
		ThrowIfFalse(prebuild_infos[i].ResultDataMaxSizeInBytes > 0);

		max_size_scratch = max(max_size_scratch, prebuild_infos[i].ScratchDataSizeInBytes);
	}

	AllocateUAVBuffer(device, max_size_scratch, &m_bottomLevelAccelerationStructure.scratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"ScratchResource");

	// Allocate resources for acceleration structures.
   // Acceleration structures can only be placed in resources that are created in the default heap (or custom heap equivalent). 
   // Default heap is OK since the application doesn�t need CPU read/write access to them. 
   // The resources that will contain acceleration structures must be created in the state D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 
   // and must have resource flag D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS. The ALLOW_UNORDERED_ACCESS requirement simply acknowledges both: 
   //  - the system will be doing this type of access in its implementation of acceleration structure builds behind the scenes.
   //  - from the app point of view, synchronization of writes/reads to acceleration structures is accomplished using UAV barriers.

	D3D12_RESOURCE_STATES initialResourceState;
	if (m_raytracingAPI == RaytracingAPI::FallbackLayer)
	{
		initialResourceState = m_fallbackDevice->GetAccelerationStructureResourceState();
	}
	else // DirectX Raytracing
	{
		initialResourceState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
	}

	for (size_t i = 0; i < num_model; i++) {
		AllocateUAVBuffer(device, prebuild_infos[i].ResultDataMaxSizeInBytes, &m_bottomLevelAccelerationStructure.structures[i], initialResourceState, L"BottomLevelAccelerationStructure");

		build_descs[i].ScratchAccelerationStructureData = m_bottomLevelAccelerationStructure.scratch->GetGPUVirtualAddress();
		build_descs[i].DestAccelerationStructureData = m_bottomLevelAccelerationStructure.structures[i]->GetGPUVirtualAddress();

		m_bottomLevelAccelerationStructure.structure_pointers[i] = CreateFallbackWrappedPointer(m_bottomLevelAccelerationStructure.structures[i], static_cast<UINT>(prebuild_infos[i].ResultDataMaxSizeInBytes) / sizeof(UINT32));
	}


	//Instance Buffer
	UINT NumInstance = 2;
	ComPtr<ID3D12Resource> instanceDescs;
	{
		vector<D3D12_RAYTRACING_FALLBACK_INSTANCE_DESC> instanceDesc;
		instanceDesc.resize(NumInstance);
		
		//plane
		instanceDesc[0].Transform[1][3] = -1;
		instanceDesc[0].Transform[0][0] = instanceDesc[0].Transform[1][1] = instanceDesc[0].Transform[2][2] = instanceDesc[0].Transform[3][3] = 3;
		instanceDesc[0].InstanceMask = 1;

		//cube
		//instanceDesc[1].Transform[1][3] = -2.5;
		instanceDesc[1].Transform[0][0] = instanceDesc[1].Transform[1][1] = instanceDesc[1].Transform[2][2] = instanceDesc[1].Transform[3][3] = 1;
		instanceDesc[1].InstanceMask = 1;

		//sphere
		//instanceDesc[2].Transform[0][3] = -3;
		//instanceDesc[2].Transform[0][0] = instanceDesc[2].Transform[1][1] = instanceDesc[2].Transform[2][2] = instanceDesc[2].Transform[3][3] = 1;
		//instanceDesc[2].InstanceMask = 1;

		////instanceDesc[3].Transform[0][3] = -3;
		//instanceDesc[3].Transform[0][0] = instanceDesc[3].Transform[1][1] = instanceDesc[3].Transform[2][2] = instanceDesc[3].Transform[3][3] = 1;
		//instanceDesc[3].InstanceMask = 1;

		//ָ��ʵ����Ӧ��BLAS
		instanceDesc[0].AccelerationStructure = m_bottomLevelAccelerationStructure.structure_pointers[ModelType::Plane];
		instanceDesc[1].AccelerationStructure = m_bottomLevelAccelerationStructure.structure_pointers[ModelType::Cube];
		/*instanceDesc[2].AccelerationStructure = m_bottomLevelAccelerationStructure.structure_pointers[ModelType::Sphere];
		instanceDesc[3].AccelerationStructure = m_bottomLevelAccelerationStructure.structure_pointers[ModelType::Cube];*/

		UINT64 bufferSize = static_cast<UINT64>(instanceDesc.size() * sizeof(instanceDesc[0]));
		AllocateUploadBuffer(device, instanceDesc.data(), bufferSize, &instanceDescs, L"InstanceDescs");
	}


	m_topLevelAccelerationStructure.structures.resize(1);
	m_topLevelAccelerationStructure.structure_pointers.resize(1);


	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc = {};
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &topLevelInputs = topLevelBuildDesc.Inputs;
	topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	topLevelInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
	topLevelInputs.NumDescs = NumInstance;//ʵ������ !!!!!!!
	topLevelInputs.pGeometryDescs = nullptr;
	topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
	topLevelInputs.InstanceDescs = instanceDescs->GetGPUVirtualAddress();

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo = {};
	m_fallbackDevice->GetRaytracingAccelerationStructurePrebuildInfo(&topLevelInputs, &topLevelPrebuildInfo);
	if (m_raytracingAPI == RaytracingAPI::FallbackLayer)
	{
		m_fallbackDevice->GetRaytracingAccelerationStructurePrebuildInfo(&topLevelInputs, &topLevelPrebuildInfo);
	}
	else // DirectX Raytracing
	{
		m_dxrDevice->GetRaytracingAccelerationStructurePrebuildInfo(&topLevelInputs, &topLevelPrebuildInfo);
	}
	ThrowIfFalse(topLevelPrebuildInfo.ResultDataMaxSizeInBytes > 0);

	AllocateUAVBuffer(device, topLevelPrebuildInfo.ResultDataMaxSizeInBytes, &m_topLevelAccelerationStructure.structures[0], initialResourceState, L"TopLevelAccelerationStructure");


	// Create a wrapped pointer to the acceleration structure.
	if (m_raytracingAPI == RaytracingAPI::FallbackLayer)
	{
		UINT numBufferElements = static_cast<UINT>(topLevelPrebuildInfo.ResultDataMaxSizeInBytes) / sizeof(UINT32);
		m_fallbackTopLevelAccelerationStructurePointer = CreateFallbackWrappedPointer(m_topLevelAccelerationStructure.structures[0], numBufferElements);
	}

	// Top Level Acceleration Structure desc
	{
		topLevelBuildDesc.DestAccelerationStructureData = m_topLevelAccelerationStructure.structures[0]->GetGPUVirtualAddress();
		topLevelBuildDesc.ScratchAccelerationStructureData = m_bottomLevelAccelerationStructure.scratch->GetGPUVirtualAddress();
		topLevelBuildDesc.Inputs.InstanceDescs = instanceDescs->GetGPUVirtualAddress();
	}

	std::vector<D3D12_RESOURCE_BARRIER> barriers;
	barriers.resize(m_bottomLevelAccelerationStructure.structures.size());
	for (size_t i = 0; i < num_model; i++) {
		barriers[i] = CD3DX12_RESOURCE_BARRIER::UAV(m_bottomLevelAccelerationStructure.structures[i]);
	}

	/*ID3D12DescriptorHeap* heaps[1] = {
	  cbv_srv_uav_heap->GetDescriptorHeap()
	};

	commandList->SetDescriptorHeaps(1, heaps);*/

	for (size_t i = 0; i < num_model; i++) {
		/*ID3D12DescriptorHeap* heaps[1] = { m_descriptorHeap.Get() };
		commandList->SetDescriptorHeaps(1, heaps);*/
		//device->PrepareCommandLists();?????
		//ID3D12DescriptorHeap* heaps[1] = { m_descriptorHeap.Get() };
		//commandList->SetDescriptorHeaps(1, heaps);
		// Set the descriptor heaps to be used during acceleration structure build for the Fallback Layer.
		ID3D12DescriptorHeap* pDescriptorHeaps[1] = { m_descriptorHeap.Get() };
		m_fallbackCommandList->SetDescriptorHeaps(1, pDescriptorHeaps);
		m_fallbackCommandList->BuildRaytracingAccelerationStructure(&build_descs[i], 0, nullptr);
		//m_deviceResources->ExecuteCommandList();
		//m_deviceResources->WaitForGpu();
	}
	commandList->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
	m_fallbackCommandList->BuildRaytracingAccelerationStructure(&topLevelBuildDesc, 0, nullptr);

	// Kick off acceleration structure construction.
	m_deviceResources->ExecuteCommandList();

	// Wait for GPU to finish as the locally created temporary GPU resources will get released once we go out of scope.
	m_deviceResources->WaitForGpu();

}

// Build shader tables.
// This encapsulates all shader records - shaders and the arguments for their local root signatures.
void D3D12RaytracingSimpleLighting::BuildShaderTables()
{
	auto device = m_deviceResources->GetD3DDevice();

	void* rayGenShaderIdentifier;
	void* missShaderIdentifier[RayType::Count];
	void* hitGroupShaderIdentifier[RayType::Count];

	auto GetShaderIdentifiers = [&](auto* stateObjectProperties)
	{
		rayGenShaderIdentifier = stateObjectProperties->GetShaderIdentifier(c_raygenShaderName);
		for (UINT i = 0; i < RayType::Count; i++)
		{
			missShaderIdentifier[i] = stateObjectProperties->GetShaderIdentifier(c_missShaderNames[i]);
			hitGroupShaderIdentifier[i] = stateObjectProperties->GetShaderIdentifier(c_hitGroupNames_TriangleGeometry[i]);
		}
	};

	// Get shader identifiers.
	UINT shaderIdentifierSize;
	if (m_raytracingAPI == RaytracingAPI::FallbackLayer)
	{
		GetShaderIdentifiers(m_fallbackStateObject.Get());
		shaderIdentifierSize = m_fallbackDevice->GetShaderIdentifierSize();
	}
	else // DirectX Raytracing
	{
		ComPtr<ID3D12StateObjectPropertiesPrototype> stateObjectProperties;
		ThrowIfFailed(m_dxrStateObject.As(&stateObjectProperties));
		GetShaderIdentifiers(stateObjectProperties.Get());
		shaderIdentifierSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
	}

	// Ray gen shader table
	{
		UINT numShaderRecords = 1;
		UINT shaderRecordSize = shaderIdentifierSize;
		ShaderTable rayGenShaderTable(device, numShaderRecords, shaderRecordSize, L"RayGenShaderTable");
		rayGenShaderTable.push_back(ShaderRecord(rayGenShaderIdentifier, shaderIdentifierSize));
		m_rayGenShaderTable = rayGenShaderTable.GetResource();
	}

	// Miss shader table
	{
		UINT numShaderRecords = RayType::Count;
		UINT shaderRecordSize = shaderIdentifierSize;
		ShaderTable missShaderTable(device, numShaderRecords, shaderRecordSize, L"MissShaderTable");
		for (UINT i = 0; i < RayType::Count; i++)
		{
			missShaderTable.push_back(ShaderRecord(missShaderIdentifier[i], shaderIdentifierSize, nullptr, 0));
		}
		m_missShaderTableStrideInBytes = missShaderTable.GetShaderRecordSize();
		m_missShaderTable = missShaderTable.GetResource();
	}

	// Hit group shader table
	{
		struct RootArguments {
			CubeConstantBuffer cb;
		} rootArguments;
		rootArguments.cb = m_cubeCB;

		UINT numShaderRecords = 2;
		UINT shaderRecordSize = shaderIdentifierSize + sizeof(rootArguments);
		ShaderTable hitGroupShaderTable(device, numShaderRecords, shaderRecordSize, L"HitGroupShaderTable");
		for (auto& hitGroupShaderID : hitGroupShaderIdentifier)
		{
			hitGroupShaderTable.push_back(ShaderRecord(hitGroupShaderID, shaderIdentifierSize, &rootArguments, sizeof(rootArguments)));
		}
		//hitGroupShaderTable.push_back(ShaderRecord(hitGroupShaderIdentifier, shaderIdentifierSize, &rootArguments, sizeof(rootArguments)));
		m_hitGroupShaderTableStrideInBytes = hitGroupShaderTable.GetShaderRecordSize();
		m_hitGroupShaderTable = hitGroupShaderTable.GetResource();
	}
}

void D3D12RaytracingSimpleLighting::SelectRaytracingAPI(RaytracingAPI type)
{
	if (type == RaytracingAPI::FallbackLayer)
	{
		m_raytracingAPI = type;
	}
	else // DirectX Raytracing
	{
		if (m_isDxrSupported)
		{
			m_raytracingAPI = type;
		}
		else
		{
			OutputDebugString(L"Invalid selection - DXR is not available.\n");
		}
	}
}

void D3D12RaytracingSimpleLighting::OnKeyDown(UINT8 key)
{
	// Store previous values.
	RaytracingAPI previousRaytracingAPI = m_raytracingAPI;
	bool previousForceComputeFallback = m_forceComputeFallback;

	float fDelta = 0.1f;

	switch (key)
	{
	case VK_NUMPAD1:
	case '1': // Fallback Layer
		m_forceComputeFallback = false;
		SelectRaytracingAPI(RaytracingAPI::FallbackLayer);
		break;
	case VK_NUMPAD2:
	case '2': // Fallback Layer + force compute path
		m_forceComputeFallback = true;
		SelectRaytracingAPI(RaytracingAPI::FallbackLayer);
		break;
	case VK_NUMPAD3:
	case '3': // DirectX Raytracing
		SelectRaytracingAPI(RaytracingAPI::DirectXRaytracing);
		break;
	case 'W':
	{
		m_eye = DirectX::XMVectorSetZ(m_eye, XMVectorGetZ(m_eye) + fDelta);
		//g_v4LightPosition.y += fDelta;
	}
	break;
	case 'S':
	{
		m_eye = DirectX::XMVectorSetZ(m_eye, XMVectorGetZ(m_eye) - fDelta);
		//g_v4LightPosition.y -= fDelta;
	}
	break;
	case VK_UP:
	{
		m_eye = DirectX::XMVectorSetY(m_eye, XMVectorGetY(m_eye) + fDelta);
		//g_v4LightPosition.y += fDelta;
	}
	break;
	case VK_DOWN:
	{
		m_eye = DirectX::XMVectorSetY(m_eye, XMVectorGetY(m_eye) - fDelta);
		//g_v4LightPosition.y -= fDelta;
	}
	break;
	case VK_LEFT:
	{
		m_eye = DirectX::XMVectorSetX(m_eye, XMVectorGetX(m_eye) - fDelta);
		//g_v4LightPosition.x -= fDelta;
	}
	break;
	case VK_RIGHT:
	{
		m_eye = DirectX::XMVectorSetX(m_eye, XMVectorGetX(m_eye) + fDelta);
		//g_v4LightPosition.x += fDelta;
	}
	default:
		break;
	}

	if (m_raytracingAPI != previousRaytracingAPI ||
		m_forceComputeFallback != previousForceComputeFallback)
	{
		// Raytracing API selection changed, recreate everything.
		RecreateD3D();
	}
}

// Update frame-based values.
void D3D12RaytracingSimpleLighting::OnUpdate()
{
	m_timer.Tick();
	CalculateFrameStats();
	float elapsedTime = static_cast<float>(m_timer.GetElapsedSeconds());
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
	auto prevFrameIndex = m_deviceResources->GetPreviousFrameIndex();

	// Rotate the camera around Y axis.
	{
		/*float secondsToRotateAround = 24.0f;
		float angleToRotateBy = 360.0f * (elapsedTime / secondsToRotateAround);
		XMMATRIX rotate = XMMatrixRotationY(XMConvertToRadians(angleToRotateBy));
		m_eye = XMVector3Transform(m_eye, rotate);
		m_up = XMVector3Transform(m_up, rotate);
		m_at = XMVector3Transform(m_at, rotate);*/
		UpdateCameraMatrices();
	}

	// Rotate the second light around Y axis.
	{
		float secondsToRotateAround = 8.0f;
		float angleToRotateBy = -360.0f * (elapsedTime / secondsToRotateAround);
		XMMATRIX rotate = XMMatrixRotationY(XMConvertToRadians(angleToRotateBy));
		const XMVECTOR& prevLightPosition = m_sceneCB[prevFrameIndex].lightPosition;
		m_sceneCB[frameIndex].lightPosition = XMVector3Transform(prevLightPosition, rotate);
	}
}


// Parse supplied command line args.
void D3D12RaytracingSimpleLighting::ParseCommandLineArgs(WCHAR* argv[], int argc)
{
	DXSample::ParseCommandLineArgs(argv, argc);

	if (argc > 1)
	{
		if (_wcsnicmp(argv[1], L"-FL", wcslen(argv[1])) == 0)
		{
			m_forceComputeFallback = true;
			m_raytracingAPI = RaytracingAPI::FallbackLayer;
		}
		else if (_wcsnicmp(argv[1], L"-DXR", wcslen(argv[1])) == 0)
		{
			m_raytracingAPI = RaytracingAPI::DirectXRaytracing;
		}
	}
}

void D3D12RaytracingSimpleLighting::DoRaytracing()
{
	auto commandList = m_deviceResources->GetCommandList();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

	auto DispatchRays = [&](auto* commandList, auto* stateObject, auto* dispatchDesc)
	{
		// Since each shader table has only one shader record, the stride is same as the size.
		dispatchDesc->HitGroupTable.StartAddress = m_hitGroupShaderTable->GetGPUVirtualAddress();
		dispatchDesc->HitGroupTable.SizeInBytes = m_hitGroupShaderTable->GetDesc().Width;
		dispatchDesc->HitGroupTable.StrideInBytes = m_hitGroupShaderTableStrideInBytes;
		dispatchDesc->MissShaderTable.StartAddress = m_missShaderTable->GetGPUVirtualAddress();
		dispatchDesc->MissShaderTable.SizeInBytes = m_missShaderTable->GetDesc().Width;
		dispatchDesc->MissShaderTable.StrideInBytes = m_missShaderTableStrideInBytes;
		dispatchDesc->RayGenerationShaderRecord.StartAddress = m_rayGenShaderTable->GetGPUVirtualAddress();
		dispatchDesc->RayGenerationShaderRecord.SizeInBytes = m_rayGenShaderTable->GetDesc().Width;
		dispatchDesc->Width = m_width;
		dispatchDesc->Height = m_height;
		dispatchDesc->Depth = 1;
		commandList->SetPipelineState1(stateObject);
		commandList->DispatchRays(dispatchDesc);
	};

	auto SetCommonPipelineState = [&](auto* descriptorSetCommandList)
	{
		descriptorSetCommandList->SetDescriptorHeaps(1, m_descriptorHeap.GetAddressOf());
		// Set index and successive vertex buffer decriptor tables
		commandList->SetComputeRootShaderResourceView(GlobalRootSignatureParams::Meshes, meshes_buffer->GetBuffer()->GetGPUVirtualAddress());
		commandList->SetComputeRootShaderResourceView(GlobalRootSignatureParams::Vertices, all_vertices_buffer->GetBuffer()->GetGPUVirtualAddress());
		commandList->SetComputeRootShaderResourceView(GlobalRootSignatureParams::Indices, all_indices_buffer->GetBuffer()->GetGPUVirtualAddress());
		commandList->SetComputeRootDescriptorTable(GlobalRootSignatureParams::OutputViewSlot, m_raytracingOutputResourceUAVGpuDescriptor);
	};

	commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());

	// Copy the updated scene constant buffer to GPU.
	memcpy(&m_mappedConstantData[frameIndex].constants, &m_sceneCB[frameIndex], sizeof(m_sceneCB[frameIndex]));
	auto cbGpuAddress = m_perFrameConstants->GetGPUVirtualAddress() + frameIndex * sizeof(m_mappedConstantData[0]);
	commandList->SetComputeRootConstantBufferView(GlobalRootSignatureParams::SceneConstantSlot, cbGpuAddress);

	// Bind the heaps, acceleration structure and dispatch rays.
	D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
	if (m_raytracingAPI == RaytracingAPI::FallbackLayer)
	{
		SetCommonPipelineState(m_fallbackCommandList.Get());
		m_fallbackCommandList->SetTopLevelAccelerationStructure(GlobalRootSignatureParams::AccelerationStructureSlot, m_fallbackTopLevelAccelerationStructurePointer);
		DispatchRays(m_fallbackCommandList.Get(), m_fallbackStateObject.Get(), &dispatchDesc);
	}
	else // DirectX Raytracing
	{
		SetCommonPipelineState(commandList);
		commandList->SetComputeRootShaderResourceView(GlobalRootSignatureParams::AccelerationStructureSlot, m_bottomLevelAccelerationStructure.structures[0]->GetGPUVirtualAddress());
		DispatchRays(m_dxrCommandList.Get(), m_dxrStateObject.Get(), &dispatchDesc);
	}
}

// Update the application state with the new resolution.
void D3D12RaytracingSimpleLighting::UpdateForSizeChange(UINT width, UINT height)
{
	DXSample::UpdateForSizeChange(width, height);
}

// Copy the raytracing output to the backbuffer.
void D3D12RaytracingSimpleLighting::CopyRaytracingOutputToBackbuffer()
{
	auto commandList = m_deviceResources->GetCommandList();
	auto renderTarget = m_deviceResources->GetRenderTarget();

	D3D12_RESOURCE_BARRIER preCopyBarriers[2];
	preCopyBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(renderTarget, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
	preCopyBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(m_raytracingOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
	commandList->ResourceBarrier(ARRAYSIZE(preCopyBarriers), preCopyBarriers);

	commandList->CopyResource(renderTarget, m_raytracingOutput.Get());

	D3D12_RESOURCE_BARRIER postCopyBarriers[2];
	postCopyBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(renderTarget, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
	postCopyBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(m_raytracingOutput.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	commandList->ResourceBarrier(ARRAYSIZE(postCopyBarriers), postCopyBarriers);
}

// Create resources that are dependent on the size of the main window.
void D3D12RaytracingSimpleLighting::CreateWindowSizeDependentResources()
{
	CreateRaytracingOutputResource();
	UpdateCameraMatrices();
}

// Release resources that are dependent on the size of the main window.
void D3D12RaytracingSimpleLighting::ReleaseWindowSizeDependentResources()
{
	m_raytracingOutput.Reset();
}

// Release all resources that depend on the device.
void D3D12RaytracingSimpleLighting::ReleaseDeviceDependentResources()
{
	m_fallbackDevice.Reset();
	m_fallbackCommandList.Reset();
	m_fallbackStateObject.Reset();
	m_raytracingGlobalRootSignature.Reset();
	m_raytracingLocalRootSignature.Reset();

	m_dxrDevice.Reset();
	m_dxrCommandList.Reset();
	m_dxrStateObject.Reset();

	m_descriptorHeap.Reset();
	m_descriptorsAllocated = 0;
	m_raytracingOutputResourceUAVDescriptorHeapIndex = UINT_MAX;
	vector<D3DBuffer>().swap(m_indexBuffer);
	vector<D3DBuffer>().swap(m_vertexBuffer);

	m_perFrameConstants.Reset();
	m_rayGenShaderTable.Reset();
	m_missShaderTable.Reset();
	m_hitGroupShaderTable.Reset();

	//RELEASE(m_bottomLevelAccelerationStructure);
	//m_topLevelAccelerationStructure

}

void D3D12RaytracingSimpleLighting::RecreateD3D()
{
	// Give GPU a chance to finish its execution in progress.
	try
	{
		m_deviceResources->WaitForGpu();
	}
	catch (HrException&)
	{
		// Do nothing, currently attached adapter is unresponsive.
	}
	m_deviceResources->HandleDeviceLost();
}

// Render the scene.
void D3D12RaytracingSimpleLighting::OnRender()
{
	if (!m_deviceResources->IsWindowVisible())
	{
		return;
	}

	m_deviceResources->Prepare();
	DoRaytracing();
	CopyRaytracingOutputToBackbuffer();

	m_deviceResources->Present(D3D12_RESOURCE_STATE_PRESENT);
}

void D3D12RaytracingSimpleLighting::OnDestroy()
{
	// Let GPU finish before releasing D3D resources.
	m_deviceResources->WaitForGpu();
	OnDeviceLost();
}

// Release all device dependent resouces when a device is lost.
void D3D12RaytracingSimpleLighting::OnDeviceLost()
{
	ReleaseWindowSizeDependentResources();
	ReleaseDeviceDependentResources();
}

// Create all device dependent resources when a device is restored.
void D3D12RaytracingSimpleLighting::OnDeviceRestored()
{
	CreateDeviceDependentResources();
	CreateWindowSizeDependentResources();
}

// Compute the average frames per second and million rays per second.
void D3D12RaytracingSimpleLighting::CalculateFrameStats()
{
	static int frameCnt = 0;
	static double elapsedTime = 0.0f;
	double totalTime = m_timer.GetTotalSeconds();
	frameCnt++;

	// Compute averages over one second period.
	if ((totalTime - elapsedTime) >= 1.0f)
	{
		float diff = static_cast<float>(totalTime - elapsedTime);
		float fps = static_cast<float>(frameCnt) / diff; // Normalize to an exact second.

		frameCnt = 0;
		elapsedTime = totalTime;

		float MRaysPerSecond = (m_width * m_height * fps) / static_cast<float>(1e6);

		wstringstream windowText;

		if (m_raytracingAPI == RaytracingAPI::FallbackLayer)
		{
			if (m_fallbackDevice->UsingRaytracingDriver())
			{
				windowText << L"(FL-DXR)";
			}
			else
			{
				windowText << L"(FL)";
			}
		}
		else
		{
			windowText << L"(DXR)";
		}
		windowText << setprecision(2) << fixed
			<< L"    fps: " << fps << L"     ~Million Primary Rays/s: " << MRaysPerSecond
			<< L"    GPU[" << m_deviceResources->GetAdapterID() << L"]: " << m_deviceResources->GetAdapterDescription();
		SetCustomWindowText(windowText.str().c_str());
	}
}

// Handle OnSizeChanged message event.
void D3D12RaytracingSimpleLighting::OnSizeChanged(UINT width, UINT height, bool minimized)
{
	if (!m_deviceResources->WindowSizeChanged(width, height, minimized))
	{
		return;
	}

	UpdateForSizeChange(width, height);

	ReleaseWindowSizeDependentResources();
	CreateWindowSizeDependentResources();
}

// Create a wrapped pointer for the Fallback Layer path.
WRAPPED_GPU_POINTER D3D12RaytracingSimpleLighting::CreateFallbackWrappedPointer(ID3D12Resource* resource, UINT bufferNumElements)
{
	auto device = m_deviceResources->GetD3DDevice();

	D3D12_UNORDERED_ACCESS_VIEW_DESC rawBufferUavDesc = {};
	rawBufferUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	rawBufferUavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
	rawBufferUavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
	rawBufferUavDesc.Buffer.NumElements = bufferNumElements;

	D3D12_CPU_DESCRIPTOR_HANDLE bottomLevelDescriptor;

	// Only compute fallback requires a valid descriptor index when creating a wrapped pointer.
	UINT descriptorHeapIndex = 0;
	if (!m_fallbackDevice->UsingRaytracingDriver())
	{
		descriptorHeapIndex = AllocateDescriptor(&bottomLevelDescriptor);
		device->CreateUnorderedAccessView(resource, nullptr, &rawBufferUavDesc, bottomLevelDescriptor);
	}
	return m_fallbackDevice->GetWrappedPointerSimple(descriptorHeapIndex, resource->GetGPUVirtualAddress());
}

std::vector<std::string> D3D12RaytracingSimpleLighting::splitStr(std::string& str, char delim) {
	std::stringstream ss(str);
	std::string token;
	std::vector<std::string> splitString;
	while (std::getline(ss, token, delim)) {
		if (token == "") {
			//If token is empty just write 0 it will result in a -1 index
			//Since that index number is nonsensical you can catch it pretty easily later
			splitString.push_back("0");
		}
		else {
			splitString.push_back(token);
		}
	}
	return splitString;
}

void D3D12RaytracingSimpleLighting::LoadMeshVertex(const CHAR * pszMeshFileName, UINT & nVertexCnt, vector<Vertex> & ppVertex, vector<Index> & ppIndices)
{
	//std::ifstream fin;
	//char input;

	//{
	//	fin.open(pszMeshFileName);
	//	if (fin.fail())
	//	{
	//		//throw CGRSCOMException(E_FAIL);
	//	}
	//	fin.get(input);
	//	while (input != ':')
	//	{
	//		fin.get(input);
	//	}
	//	fin >> nVertexCnt;

	//	fin.get(input);
	//	while (input != ':')
	//	{
	//		fin.get(input);
	//	}
	//	fin.get(input);
	//	fin.get(input);

	//	ppVertex.resize(nVertexCnt);
	//	ppIndices.resize(nVertexCnt);

	//	for (UINT i = 0; i < nVertexCnt; i++)
	//	{
	//		fin >> ppVertex[i].position.x >> ppVertex[i].position.y >> ppVertex[i].position.z;
	//		//ppVertex[i].position.w = 1.0f;
	//		fin >> ppVertex[i].texture.x >> ppVertex[i].texture.y;
	//		fin >> ppVertex[i].normal.x >> ppVertex[i].normal.y >> ppVertex[i].normal.z;

	//		ppIndices[i] = static_cast<Index>(i);
	//	}
	//}

	std::ifstream file;
	char input;
	file.open(pszMeshFileName);
	std::string line, key, x, y, z;
	vector<XMFLOAT3> pos, normal, texture;
	char delimeter = '/';
	uint32_t index = 0;

	ppVertex.resize(0);
	ppIndices.resize(0);
	while (!file.eof()) {
		std::getline(file, line);
		std::istringstream iss(line);
		iss >> key;
		if (key == "v") { //Vertex data
			iss >> x >> y >> z;
			XMFLOAT3 temp(std::stof(x), std::stof(y), std::stof(z));
			pos.push_back(temp);
		}
		else if (key == "vn") { //Normal data
			iss >> x >> y >> z;
			XMFLOAT3 temp(std::stof(x), std::stof(y), std::stof(z));
			normal.push_back(temp);
		}
		else if (key == "vt") { //Texture data
			iss >> x >> y;
			XMFLOAT3 temp(std::stof(x), std::stof(y), std::stof(z));
			texture.push_back(temp);
		}
		else if (key == "f") { //index data  v/vt/vn
			iss >> x >> y >> z;
			std::vector<std::string> splitX = splitStr(x, delimeter);
			std::vector<std::string> splitY = splitStr(y, delimeter);
			std::vector<std::string> splitZ = splitStr(z, delimeter);
			vector<vector<std::string>> m({ splitX , splitY, splitZ});
			for (int i = 0; i < 3; ++i) {
				Vertex temp;
				temp.position = pos[stoi(m[i][0])-1];
				temp.normal = normal[stoi(m[i][2])-1];
				temp.texture = texture[stoi(m[i][1])-1];
				ppVertex.push_back(temp);
				ppIndices.push_back(static_cast<Index>(index++));
			}
		}
		nVertexCnt = index;
	}
}

// Allocate a descriptor and return its index. 
// If the passed descriptorIndexToUse is valid, it will be used instead of allocating a new one.
UINT D3D12RaytracingSimpleLighting::AllocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE* cpuDescriptor, UINT descriptorIndexToUse)
{
	auto descriptorHeapCpuBase = m_descriptorHeap->GetCPUDescriptorHandleForHeapStart();
	if (descriptorIndexToUse >= m_descriptorHeap->GetDesc().NumDescriptors)
	{
		descriptorIndexToUse = m_descriptorsAllocated++;
	}
	*cpuDescriptor = CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeapCpuBase, descriptorIndexToUse, m_descriptorSize);
	return descriptorIndexToUse;
}

// Create SRV for a buffer.
UINT D3D12RaytracingSimpleLighting::CreateBufferSRV(D3DBuffer* buffer, UINT numElements, UINT elementSize)
{
	auto device = m_deviceResources->GetD3DDevice();

	// SRV
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Buffer.NumElements = numElements;
	if (elementSize == 0)
	{
		srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
		srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
		srvDesc.Buffer.StructureByteStride = 0;
	}
	else
	{
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
		srvDesc.Buffer.StructureByteStride = elementSize;
	}
	UINT descriptorIndex = AllocateDescriptor(&buffer->cpuDescriptorHandle);
	device->CreateShaderResourceView(buffer->resource.Get(), &srvDesc, buffer->cpuDescriptorHandle);
	buffer->gpuDescriptorHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(m_descriptorHeap->GetGPUDescriptorHandleForHeapStart(), descriptorIndex, m_descriptorSize);
	return descriptorIndex;
};