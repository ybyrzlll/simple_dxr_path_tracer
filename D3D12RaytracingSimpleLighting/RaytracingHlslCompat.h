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

#define CPP_REGISTER_MATERIALS 4
#define HLSL_REGISTER_MATERIALS t4

#endif

// PERFORMANCE TIP: Set max recursion depth as low as needed
// as drivers may apply optimization strategies for low recursion depths.
#define MAX_RAY_RECURSION_DEPTH 3    // ~ primary rays + reflections + shadow rays from reflected geometry.

#define Sample_Num 5

#define PI 3.1415926535

#define rayTmin = 1e-4f;
#define rayTmax = 1e27f;

// Shader will use byte encoding to access indices.
typedef uint32_t Index;

// Ray types traced in this sample.
namespace ModelType {
	enum Enum {
		Light,
		Plane,
		Sphere
	};
}

struct ProceduralPrimitiveAttributes
{
	XMFLOAT3 normal;
};

struct Mesh
{
	UINT first_idx_vertices;
	UINT first_idx_indices;
	UINT material;
};

struct SceneConstantBuffer
{
    XMMATRIX projectionToWorld;
    XMVECTOR cameraPosition;
    XMVECTOR lightPosition;
    XMVECTOR lightColor;
    XMVECTOR lightDiffuseColor;
	UINT frame_num;
};

struct MaterialConstantBuffer
{
	XMFLOAT4 albedo;
	float roughness;
	float metallic;
	float specular;

	float reflectanceCoef;
	float diffuseCoef;
	float specularCoef;
	float specularPower;
	float stepScale;                      // Step scale for ray marching of signed distance primitives. 
										  // - Some object transformations don't preserve the distances and 
										  //   thus require shorter steps.
	XMFLOAT3 padding;
};

struct Material
{
	XMFLOAT4 color_emissive;
	XMFLOAT4 color_ambient;
	XMFLOAT4 color_diffuse;
	XMFLOAT4 color_specular;
	float roughness;
	float metallic;
	float specular;
	XMFLOAT4  emission;
};

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT3 texture;
	XMFLOAT3 normal;
};


struct RayPayload
{
	UINT   recursionDepth;
	XMFLOAT3 attenuation;
	XMFLOAT4 radiance;
	UINT seed;
};

struct ShadowRayPayload
{
	bool hit;
};

struct Ray
{
	XMFLOAT3 origin;
	XMFLOAT3 direction;
};


// Ray types traced in this sample.
namespace RayType {
	enum Enum {
		Radiance = 0,   // ~ Primary, reflected camera/view rays calculating color for each hit.
		Shadow,         // ~ Shadow/visibility rays, only testing for occlusion
		Count
	};
}

namespace TraceRayParameters
{
	static const UINT InstanceMask = ~0;   // Everything is visible.
	namespace HitGroup {
		static const UINT Offset[RayType::Count] =
		{
			0, // Radiance ray
			1  // Shadow ray
		};
		static const UINT GeometryStride = RayType::Count;
	}
	namespace MissShader {
		static const UINT Offset[RayType::Count] =
		{
			0, // Radiance ray
			1  // Shadow ray
		};
	}
}

static const float InShadowRadiance = 0.35f;
static const XMFLOAT4 BackgroundColor = XMFLOAT4(0.2f, 0.2f, 0.2f, 1.0f);

#endif // RAYTRACINGHLSLCOMPAT_H