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

#ifndef RAYTRACINGHLSLCOMPAT_H
#define RAYTRACINGHLSLCOMPAT_H

#ifdef HLSL
#include "HlslCompat.h"
#else
using namespace DirectX;

#define CPP_REGISTER_MESHES 1
#define HLSL_REGISTER_MESHES t1

#define CPP_REGISTER_VERTICES 2
#define HLSL_REGISTER_VERTICES t2

#define CPP_REGISTER_INDICES 3
#define HLSL_REGISTER_INDICES t3


// Shader will use byte encoding to access indices.
typedef UINT16 Index;
#endif

struct Mesh
{
	UINT first_idx_vertices;
	UINT first_idx_indices;
	//UINT material;
};

struct SceneConstantBuffer
{
    XMMATRIX projectionToWorld;
    XMVECTOR cameraPosition;
    XMVECTOR lightPosition;
    XMVECTOR lightAmbientColor;
    XMVECTOR lightDiffuseColor;
};

struct CubeConstantBuffer
{
    XMFLOAT4 albedo;
	float reflectanceCoef;
	float diffuseCoef;
	float specularCoef;
	float specularPower;
	float stepScale;                      // Step scale for ray marching of signed distance primitives. 
										  // - Some object transformations don't preserve the distances and 
										  //   thus require shorter steps.
	XMFLOAT3 padding;
};

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT3 texture;
	XMFLOAT3 normal;
};

static const float InShadowRadiance = 0.35f;

#endif // RAYTRACINGHLSLCOMPAT_H