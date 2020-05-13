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

#ifndef RAYTRACING_HLSL
#define RAYTRACING_HLSL

#define HLSL
#include "RaytracingHlslCompat.h"
#include "util.hlsli"
#include "model.hlsli"
#include "GetShadingData.hlsli"
#include "sampling.hlsli"

RaytracingAccelerationStructure Scene : register(t0, space0);
RWTexture2D<float4> RenderTarget : register(u0);

ConstantBuffer<SceneConstantBuffer> g_sceneCB : register(b0);


struct RayPayload
{
	float3 radiance;
	float3 attenuation;
	float3 hitPos;
	float3 bounceDir;
	//uint terminateRay;		
	uint rayDepth;
	uint seed;
};

struct ShadowRayPayload
{
	bool hit;
};

void samplingBRDF(out float3 sampleDir, out float sampleProb, out float3 brdfCos,
	in float3 surfaceNormal, in float3 baseDir, in Material mtl, inout uint seed)
{

	float3 brdfEval;
	float3 albedo = mtl.color_diffuse;
	//uint reflectType = mtl.type;

	float3 I, O = baseDir, N = surfaceNormal, H;
	float ON = dot(O, N), IN, HN, OH;
	float alpha2 = mtl.roughness * mtl.roughness;

	//if (reflectType == Lambertian)
	/*{
		I = sample_hemisphere_cos(seed);
		IN = I.z;
		I = applyRotationMappingZToN(N, I);

		sampleProb = InvPi * IN;
		brdfEval = InvPi * albedo;
	}*/

	//else if (reflectType == Metal)
	//{
	//	H = sample_hemisphere_TrowbridgeReitzCos(alpha2, seed);
	//	HN = H.z;
	//	H = applyRotationMappingZToN(N, H);
	//	OH = dot(O, H);

	//	I = 2 * OH * H - O;
	//	IN = dot(I, N);

	//	if (IN < 0)
	//	{
	//		brdfEval = 0;
	//		sampleProb = 0;		// sampleProb = D*HN / (4*abs(OH));  if allowing sample negative hemisphere
	//	}
	//	else
	//	{
	//		float D = TrowbridgeReitz(HN*HN, alpha2);
	//		float G = Smith_TrowbridgeReitz(I, O, H, N, alpha2);
	//		float3 F = albedo + (1 - albedo) * pow(max(0, 1 - OH), 5);
	//		brdfEval = ((D * G) / (4 * IN * ON)) * F;
	//		sampleProb = D * HN / (4 * OH);		// IN > 0 imply OH > 0
	//	}
	//}

	//else if (reflectType == Plastic)
	{
		/*float metallic;
		float specular;*/
		float r = mtl.specular;

		if (rnd(seed) < r)
		{
			H = sample_hemisphere_TrowbridgeReitzCos(alpha2, seed);
			HN = H.z;
			H = applyRotationMappingZToN(N, H);
			OH = dot(O, H);

			I = 2 * OH * H - O;
			IN = dot(I, N);
		}
		else
		{
			I = sample_hemisphere_cos(seed);
			IN = I.z;
			I = applyRotationMappingZToN(N, I);

			H = O + I;
			H = (1 / length(H)) * H;
			HN = dot(H, N);
			OH = dot(O, H);
		}

		if (IN < 0)
		{
			brdfEval = 0;
			sampleProb = 0;		//sampleProb = r * (D*HN / (4*abs(OH)));  if allowing sample negative hemisphere
		}
		else
		{
			float D = TrowbridgeReitz(HN*HN, alpha2);
			float G = Smith_TrowbridgeReitz(I, O, H, N, alpha2);
			float3 spec = ((D * G) / (4 * IN * ON));
			brdfEval = r * spec + (1 - r) * InvPi * albedo;
			sampleProb = r * (D*HN / (4 * OH)) + (1 - r) * (InvPi * IN);
		}
	}

	sampleDir = I;
	brdfCos = brdfEval * IN;
}


//***************************************************************************
//****************------ Utility functions -------***************************
//***************************************************************************
// Retrieve hit world position.
float3 HitWorldPosition()
{
	return WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
}

// Diffuse lighting calculation.
float CalculateDiffuseCoefficient(in float3 hitPosition, in float3 incidentLightRay, in float3 normal)
{
	float fNDotL = saturate(dot(-incidentLightRay, normal));
	return fNDotL;
}

// Phong lighting specular component
float4 CalculateSpecularCoefficient(in float3 hitPosition, in float3 incidentLightRay, in float3 normal, in float specularPower)
{
	float3 reflectedLightRay = normalize(reflect(incidentLightRay, normal));
	return pow(saturate(dot(reflectedLightRay, normalize(-WorldRayDirection()))), specularPower);
}


typedef BuiltInTriangleIntersectionAttributes MyAttributes;


// Retrieve attribute at a hit position interpolated from vertex attributes using the hit's barycentrics.
float3 HitAttribute(float3 vertexAttribute[3], BuiltInTriangleIntersectionAttributes attr)
{
    return vertexAttribute[0] +
        attr.barycentrics.x * (vertexAttribute[1] - vertexAttribute[0]) +
        attr.barycentrics.y * (vertexAttribute[2] - vertexAttribute[0]);
}

//***************************************************************************
//*****------ TraceRay wrappers for radiance and shadow rays. -------********
//***************************************************************************

float3 tracePath(in float3 startPos, in float3 startDir, inout uint seed)
{
	float3 radiance = 0.0f;
	float3 attenuation = 1.0f;
	float rayTmin = 1e-4f;
	float rayTmax = 1e27f;


	RayDesc ray = Ray(startPos, startDir, rayTmin, rayTmax);
	RayPayload prd;
	prd.seed = seed;
	prd.rayDepth = 0;
	//prd.terminateRay = false;

	while(prd.rayDepth < MAX_RAY_RECURSION_DEPTH)
	{
		TraceRay(Scene, 0, ~0, 0, 1, 0, ray, prd);
	
		radiance += attenuation * prd.radiance;
		attenuation *= prd.attenuation;

		/*if(prd.terminateRay)
			break;*/
	
		ray.Origin = prd.hitPos;
		ray.Direction = prd.bounceDir;
		++prd.rayDepth;
	}
	
	seed = prd.seed;

	return radiance;
}


//***************************************************************************
//********************------ Ray gen shader.. -------************************
//***************************************************************************

// Generate a ray in world space for a camera pixel corresponding to an index from the dispatched 2D grid.
inline void GenerateCameraRay(float2 index, out float3 origin, out float3 direction)
{
    float2 xy = index + 0.5f; // center in the middle of the pixel.
    float2 screenPos = xy / DispatchRaysDimensions().xy * 2.0 - 1.0;

    // Invert Y for DirectX-style coordinates.
    screenPos.y = -screenPos.y;

    // Unproject the pixel coordinate into a ray.
    float4 world = mul(float4(screenPos, 0, 1), g_sceneCB.projectionToWorld);

    world.xyz /= world.w;
    origin = g_sceneCB.cameraPosition.xyz;
    direction = normalize(world.xyz - origin);
}

[shader("raygeneration")]
void MyRaygenShader()
{
    float3 rayDir;
    float3 origin;
    
	float4 color = { 0,0,0,0 };

	/*uint2 launchIdx = DispatchRaysIndex().xy;
	uint2 launchDim = DispatchRaysDimensions().xy;
	uint bufferOffset = launchDim.x * launchIdx.y + launchIdx.x;*/
	uint seed = getNewSeed(g_sceneCB.frame_num+1, g_sceneCB.frame_num, 8);

	float3 newRadiance = 0.0f;

    // Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
	[unroll]
	for (int i = 0; i < Sample_Num; i++) {
		float2 random = float2(rnd(seed), rnd(seed));
		GenerateCameraRay((float2)DispatchRaysIndex().xy + random, origin, rayDir);
		//GenerateCameraRay((float2)DispatchRaysIndex().xy , origin, rayDir);
		
		newRadiance += tracePath(origin, rayDir, seed);
	}
	newRadiance *= 1.0f / float(Sample_Num);

	float3 avrRadiance;
	if (g_sceneCB.frame_num == 0)
		avrRadiance = newRadiance;
	else
		avrRadiance = lerp(RenderTarget[DispatchRaysIndex().xy].xyz, newRadiance, 1.f / (g_sceneCB.frame_num + 1.0f));

    RenderTarget[DispatchRaysIndex().xy] = float4(avrRadiance, 1.0f);
}


[shader("closesthit")]
void MyClosestHitShader(inout RayPayload payload, in MyAttributes attr)
{
	ShadingData hit = GetShadingData(attr);

	float3 N = hit.normal, fN, E = -WorldRayDirection();
	//computeNormal(N, fN, attr);
	float EN = dot(E, N);// , EfN = dot(E, fN);

	payload.attenuation = 1.0f;


	/*if (obj.twoSided && EfN < 0)
	{
		mtlIdx = obj.backMaterialIdx;
		N = -N;
		EN = -EN;
	}*/

	if (EN < 0)
	{
		/*payload.bounceDir = WorldRayDirection();
		--payload.rayDepth;*/
		//Ray sampleRay = { HitWorldPosition(), WorldRayDirection() };
		//payload.color =  TraceRadianceRay(sampleRay, payload.recursionDepth, payload.seed, payload.attenuation);
		return;
	}

	if (any(hit.material.emission)) //光源
	{
		payload.radiance = hit.material.emission;
		return;
	}

	float3 sampleDir, brdfCos;
	float sampleProb;
	samplingBRDF(sampleDir, sampleProb, brdfCos, N, E, hit.material, payload.seed);

	if (dot(sampleDir, N) <= 0) {
		//Stop! ==没有折射
		return; // payload.rayDepth = maxPathLength;
	}
		
	//Ray sampleRay = { HitWorldPosition(), sampleDir };
	//payload.attenuation = brdfCos / sampleProb;
	//float color = payload.attenuation * float4(brdfCos / sampleProb, 1);// brdfCos/sampleProb
	//payload.attenuation *= payload.attenuation;

	////float4 sampleColor = TraceRadianceRay(sampleRay, payload.recursionDepth, payload.seed, payload.attenuation);
	//payload.color = color;// +sampleColor;
	////payload.bounceDir = sampleDir;

	////]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

	
	payload.attenuation = brdfCos / sampleProb;
	payload.bounceDir = sampleDir;

}

[shader("miss")]
void MyMissShader(inout RayPayload payload)
{
	payload.radiance = BackgroundColor;
	payload.rayDepth = MAX_RAY_RECURSION_DEPTH;
}

[shader("miss")]
void MyMissShader_ShadowRay(inout ShadowRayPayload rayPayload)
{
	rayPayload.hit = false;
}

#endif // RAYTRACING_HLSL