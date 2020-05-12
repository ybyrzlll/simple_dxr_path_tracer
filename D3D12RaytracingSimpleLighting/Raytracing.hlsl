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


// Phong lighting model = ambient + diffuse + specular components.
//float4 CalculatePhongLighting(in float4 albedo, in float3 normal, in bool isInShadow, in float diffuseCoef = 1.0, in float specularCoef = 1.0, in float specularPower = 50)
//{
//	float3 hitPosition = HitWorldPosition();
//	float3 lightPosition = g_sceneCB.lightPosition.xyz;
//	float shadowFactor = isInShadow ? InShadowRadiance : 1.0;
//	float3 incidentLightRay = normalize(hitPosition - lightPosition);
//
//	// Diffuse component.
//	float4 lightDiffuseColor = g_sceneCB.lightDiffuseColor;
//	float Kd = CalculateDiffuseCoefficient(hitPosition, incidentLightRay, normal);
//	float4 diffuseColor = shadowFactor * diffuseCoef * Kd * lightDiffuseColor * albedo;
//
//	// Specular component.
//	float4 specularColor = float4(0, 0, 0, 0);
//	if (!isInShadow)
//	{
//		float4 lightSpecularColor = float4(1, 1, 1, 1);
//		float4 Ks = CalculateSpecularCoefficient(hitPosition, incidentLightRay, normal, specularPower);
//		specularColor = specularCoef * Ks * lightSpecularColor;
//	}
//
//	// Ambient component.
//	// Fake AO: Darken faces with normal facing downwards/away from the sky a little bit.
//	float4 ambientColor = g_sceneCB.lightAmbientColor;
//	float4 ambientColorMin = g_sceneCB.lightAmbientColor - 0.1;
//	float4 ambientColorMax = g_sceneCB.lightAmbientColor;
//	float a = 1 - saturate(dot(normal, float3(0, -1, 0)));
//	ambientColor = albedo * lerp(ambientColorMin, ambientColorMax, a);
//
//	return ambientColor + diffuseColor + specularColor;
//}

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
// Trace a radiance ray into the scene and returns a shaded color.
float4 TraceRadianceRay(in Ray ray, in UINT currentRayRecursionDepth, in UINT seed, in float3 attenuation)
{
	if (currentRayRecursionDepth >= MAX_RAY_RECURSION_DEPTH)//MAX_RAY_RECURSION_DEPTH
	{
		return float4(0, 0, 0, 0);
	}

	// Set the ray's extents.
	RayDesc rayDesc;
	rayDesc.Origin = ray.origin;
	rayDesc.Direction = ray.direction;
	// Set TMin to a zero value to avoid aliasing artifacts along contact areas.
	// Note: make sure to enable face culling so as to avoid surface face fighting.
	rayDesc.TMin = 0;
	rayDesc.TMax = 10000;
	RayPayload rayPayload = { float4(0, 0, 0, 0), currentRayRecursionDepth + 1 , seed, attenuation};
	//TraceRay(Scene,
	//	RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
	//	~0, // instance mask
	//	0, // hitgroup index
	//	1, // geom multiplier
	//	0, // miss index
	//	rayDesc, rayPayload);
	TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, rayDesc, rayPayload);

	return rayPayload.color;
}

// Trace a shadow ray and return true if it hits any geometry.
bool TraceShadowRayAndReportIfHit(in Ray ray, in UINT currentRayRecursionDepth)
{
	if (currentRayRecursionDepth >= MAX_RAY_RECURSION_DEPTH)
	{
		return false;
	}

	// Set the ray's extents.
	RayDesc rayDesc;
	rayDesc.Origin = ray.origin;
	rayDesc.Direction = ray.direction;
	// Set TMin to a zero value to avoid aliasing artifcats along contact areas.
	// Note: make sure to enable back-face culling so as to avoid surface face fighting.
	rayDesc.TMin = 0;
	rayDesc.TMax = 10000;

	// Initialize shadow ray payload.
	// Set the initial value to true since closest and any hit shaders are skipped. 
	// Shadow miss shader, if called, will set it to false.
	ShadowRayPayload shadowPayload = { true };
	TraceRay(Scene,
		RAY_FLAG_CULL_BACK_FACING_TRIANGLES
		| RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
		| RAY_FLAG_FORCE_OPAQUE             // ~skip any hit shaders
		| RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, // ~skip closest hit shaders,
		TraceRayParameters::InstanceMask,
		TraceRayParameters::HitGroup::Offset[RayType::Shadow],
		TraceRayParameters::HitGroup::GeometryStride,
		TraceRayParameters::MissShader::Offset[RayType::Shadow],
		rayDesc, shadowPayload);

	return shadowPayload.hit;
}

//***************************************************************************
//********************------ Diffuse Shade.. -------*************************
//***************************************************************************

float4 DiffuseShade(in float3 DiffuseColor, in float Roughness, in float3 L, in float3 V, in float3 N) {
	float NoL = saturate(dot(N, L));
	float NoV = saturate(dot(N, V));
	float VoL = saturate(dot(V, L));
	float InvLenH = rsqrt(2 + 2 * VoL);
	float NoH = saturate((NoL + NoV) * InvLenH);
	float VoH = saturate(InvLenH + InvLenH * VoL); //saturate(dot(N, normalize(N+L)));//
	return float4(Diffuse_OrenNayar(DiffuseColor, Roughness, NoV, NoL, VoH), 1.0);
	//return float4(Diffuse_Burley(DiffuseColor, g_cubeCB.roughness, NoV, NoL, VoH), 1.0);
}

//***************************************************************************
//********************------ Specular Shade.. -------*************************
//***************************************************************************

float4 SpecularShade(in float3 SpecColor, in float Roughness, in float3 L, in float3 V, in float3 N) {
	float NoL = dot(N, L);
	float NoV = dot(N, V);
	float VoL = dot(V, L);
	float InvLenH = rsqrt(2 + 2 * VoL);
	float NoH = saturate((NoL + NoV) * InvLenH);
	float VoH = saturate(InvLenH + InvLenH * VoL); //saturate(dot(N, normalize(N+L)));//
	return float4(Cook_Torrance2(g_cubeCB.albedo.xyz, g_cubeCB.roughness, NoV, NoL, VoH, NoH), 1.0);
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

// Diffuse lighting calculation.
float4 CalculateDiffuseLighting(float3 hitPosition, float3 normal)
{
    float3 pixelToLight = normalize(g_sceneCB.lightPosition.xyz - hitPosition);

    // Diffuse contribution.
    float fNDotL = max(0.0f, dot(pixelToLight, normal));

    return g_cubeCB.albedo * g_sceneCB.lightDiffuseColor * fNDotL;
}

[shader("raygeneration")]
void MyRaygenShader()
{
    float3 rayDir;
    float3 origin;
    
	float4 color = { 0,0,0,0 };

	uint seed = initRand(g_sceneCB.frame_num, g_sceneCB.frame_num+1, 16);

    // Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
	[unroll]
	for (int i = 0; i < Sample_Num; i++) {
		/*float2 random = float2(nextRand(seed), nextRand(seed));
		GenerateCameraRay((float2)DispatchRaysIndex().xy + random, origin, rayDir);*/
		GenerateCameraRay((float2)DispatchRaysIndex().xy , origin, rayDir);
		Ray ray;
		ray.origin = origin;
		ray.direction = rayDir;
		color += TraceRadianceRay(ray, 0, seed, 1.0f);
	}
    // Write the raytraced color to the output texture.
    RenderTarget[DispatchRaysIndex().xy] = color / Sample_Num;
}


[shader("closesthit")]
void MyClosestHitShader(inout RayPayload payload, in MyAttributes attr)
{
	ShadingData hit = GetShadingData(attr);
	if (any(hit.material.emission)) //光源
	{
		payload.color = hit.material.emission;
		return;
	}
	payload.color = float4(0, 1, 0, 1.0);
	//[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[

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
		return;
	}

	/*Material mtl = materialBuffer[mtlIdx];

	if (any(mtl.emittance))
	{
		payload.radiance += mtl.emittance;
	}*/

	float3 sampleDir, brdfCos;
	float sampleProb;
	samplingBRDF(sampleDir, sampleProb, brdfCos, N, E, hit.material, payload.seed);

	if (dot(sampleDir, N) <= 0) {
		//Stop! ==没有折射
		return; // payload.rayDepth = maxPathLength;
	}
		
	Ray sampleRay = { HitWorldPosition(), sampleDir };
	payload.attenuation = brdfCos / sampleProb;
	float color = payload.attenuation * float4(brdfCos / sampleProb, 1);// brdfCos/sampleProb
	payload.attenuation *= payload.attenuation;

	float4 sampleColor = TraceRadianceRay(sampleRay, payload.recursionDepth, payload.seed, payload.attenuation);
	payload.color = color;// +sampleColor;
	//payload.bounceDir = sampleDir;

	//]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]


	//float3 hitPosition = HitWorldPosition() + hit.normal * 0.05;
	//float3 Hit2Light = normalize(g_sceneCB.lightPosition.xyz - hitPosition);
	//Ray shadowRay = { hitPosition,  Hit2Light };
	//bool shadowRayHit = TraceShadowRayAndReportIfHit(shadowRay, payload.recursionDepth);
	//float3 L = WorldRayDirection();
	//float3 V = Hit2Light;
	//float3 N = hit.normal;

	//float4 fresnelR = float4(F_Schlick2(g_cubeCB.albedo.xyz, V, L), 1.0);

	//// Reflected component.
	//float4 reflectedColor = float4(0, 0, 0, 0);
	////if (l_materialCB.reflectanceCoef > 0.001)
	//{
	//	//Trace a reflection ray.
	//	Ray reflectionRay = { hitPosition, reflect(WorldRayDirection(), hit.normal) };
	//	float4 reflectionColor = TraceRadianceRay(reflectionRay, payload.recursionDepth);
	//	reflectedColor = 0.3 * fresnelR * reflectionColor;
	//}

	////float4 phongColor = CalculatePhongLighting(g_cubeCB.albedo, hit.normal, shadowRayHit, g_cubeCB.diffuseCoef, g_cubeCB.specularCoef, g_cubeCB.specularPower);
	//
	//float4 ambient = float4(0.1, 0.1, 0.1, 1.0);
	//
	//

	//float4 DiffuseColor = DiffuseShade(g_cubeCB.albedo.xyz, g_cubeCB.roughness, L, V, N);
	//
	//float4 SpecularColor = float4(0, 0, 0, 0);

	///*if (!shadowRayHit) {
	//	payload.color += disneyDiColor;
	//}*/
	//SpecularColor = SpecularShade(g_cubeCB.albedo.xyz, g_cubeCB.roughness, -L, V, N);

	//// Apply visibility falloff.
	////float t = RayTCurrent();
	////color = lerp(color, BackgroundColor, 1.0 - exp(-0.000002*t*t*t));

	////(1- fresnelR)* DiffuseColor + fresnelR * CookTorranceColor + reflectedColor
	//float4 res = float4(0, 0, 0, 0); 
	////res += saturate(DiffuseColor);// *(1 - fresnelR);//*(1- fresnelR) 
	////res += saturate(SpecularColor);// *fresnelR;
	////res += reflectedColor;
	//res += CalculateDiffuseLighting(HitWorldPosition(), hit.normal);
	//payload.color = res;// CookTorranceColor;//CalculateDiffuseLighting(HitWorldPosition(), hit.normal); //);//color;
}

[shader("miss")]
void MyMissShader(inout RayPayload payload)
{
    payload.color = BackgroundColor;
}

[shader("miss")]
void MyMissShader_ShadowRay(inout ShadowRayPayload rayPayload)
{
	rayPayload.hit = false;
}

#endif // RAYTRACING_HLSL