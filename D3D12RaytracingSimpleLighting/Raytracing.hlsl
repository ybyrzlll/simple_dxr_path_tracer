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

RaytracingAccelerationStructure Scene : register(t0, space0);
RWTexture2D<float4> RenderTarget : register(u0);
StructuredBuffer<Mesh> scene_meshes : register(t1);//HLSL_REGISTER_MESHES
StructuredBuffer<Vertex> Vertices : register(t2);//HLSL_REGISTER_VERTICES
StructuredBuffer<uint> Indices : register(t3); //HLSL_REGISTER_INDICES

ConstantBuffer<SceneConstantBuffer> g_sceneCB : register(b0);
ConstantBuffer<MaterialConstantBuffer> g_cubeCB : register(b1);

//********************************************************************************
//******************************** GetShadingData ********************************
//********************************************************************************
typedef BuiltInTriangleIntersectionAttributes TriangleAttributes;

struct InterpolatedVertex
{
	float3 position;
	float3 normal;
	/*float3 tangent;
	float3 bitangent;
	float2 uv;
	float4 color;*/
};

struct Triangle
{
	Vertex vertices[3];
};

uint3 GetIndices()
{
	int prim_idx = PrimitiveIndex();
	int mesh_idx = InstanceID();

	return uint3(
		scene_meshes[mesh_idx].first_idx_vertices + Indices[scene_meshes[mesh_idx].first_idx_indices + (prim_idx * 3) + 0],
		scene_meshes[mesh_idx].first_idx_vertices + Indices[scene_meshes[mesh_idx].first_idx_indices + (prim_idx * 3) + 1],
		scene_meshes[mesh_idx].first_idx_vertices + Indices[scene_meshes[mesh_idx].first_idx_indices + (prim_idx * 3) + 2]
		);
}

Triangle GetTriangle()
{
	uint3 indices = GetIndices();

	Triangle tri;
	tri.vertices[0] = Vertices[indices.x];
	tri.vertices[1] = Vertices[indices.y];
	tri.vertices[2] = Vertices[indices.z];

	return tri;
}

InterpolatedVertex CalculateInterpolatedVertex(in Vertex v[3], in float2 barycentrics)
{
	float3 bary_factors = CalculateBarycentricalInterpolationFactors(barycentrics);

	InterpolatedVertex vertex;
	vertex.position = BarycentricInterpolation(v[0].position, v[1].position, v[2].position, bary_factors);
	vertex.normal = normalize(BarycentricInterpolation(v[0].normal, v[1].normal, v[2].normal, bary_factors));
	/*vertex.tangent = normalize(BarycentricInterpolation(v[0].tangent, v[1].tangent, v[2].tangent, bary_factors));
	vertex.bitangent = normalize(cross(vertex.normal, vertex.tangent));
	vertex.uv = BarycentricInterpolation(v[0].uv, v[1].uv, v[2].uv, bary_factors);
	vertex.color = BarycentricInterpolation(v[0].color, v[1].color, v[2].color, bary_factors);*/

	return vertex;
}

struct ShadingData
{
	//uint shading_model;
	float3 position;
	float3 normal;
	/*float3 diffuse;
	float3 emissive;
	float index_of_refraction;
	float glossiness;*/
};

//inline float4 SampleTexture(in SamplerState samplr, in Texture2D tex, in float2 uv)
//{
//	return tex.SampleLevel(samplr, uv, 0, 0);
//}

inline ShadingData GetShadingData(TriangleAttributes attr)
{
	ShadingData data;

	Triangle tri = GetTriangle();
	InterpolatedVertex vertex = CalculateInterpolatedVertex(tri.vertices, attr.barycentrics);
	//Material material = scene_materials[scene_meshes[InstanceID()].material];

	//data.shading_model = material.shading_model;
	data.position = WorldRayOrigin() + (WorldRayDirection() * RayTCurrent());
	data.normal = normalize(vertex.normal);
	/*data.diffuse = material.diffuse_map != MATERIAL_NO_TEXTURE_INDEX ? SampleTexture(scene_sampler, scene_textures[material.diffuse_map], vertex.uv).xyz : material.color_diffuse.xyz;
	data.emissive = material.emissive_map != MATERIAL_NO_TEXTURE_INDEX ? SampleTexture(scene_sampler, scene_textures[material.emissive_map], vertex.uv).xyz : material.color_emissive.xyz;
	data.index_of_refraction = material.index_of_refraction;
	data.glossiness = material.glossiness;*/

	return data;
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
float4 CalculatePhongLighting(in float4 albedo, in float3 normal, in bool isInShadow, in float diffuseCoef = 1.0, in float specularCoef = 1.0, in float specularPower = 50)
{
	float3 hitPosition = HitWorldPosition();
	float3 lightPosition = g_sceneCB.lightPosition.xyz;
	float shadowFactor = isInShadow ? InShadowRadiance : 1.0;
	float3 incidentLightRay = normalize(hitPosition - lightPosition);

	// Diffuse component.
	float4 lightDiffuseColor = g_sceneCB.lightDiffuseColor;
	float Kd = CalculateDiffuseCoefficient(hitPosition, incidentLightRay, normal);
	float4 diffuseColor = shadowFactor * diffuseCoef * Kd * lightDiffuseColor * albedo;

	// Specular component.
	float4 specularColor = float4(0, 0, 0, 0);
	if (!isInShadow)
	{
		float4 lightSpecularColor = float4(1, 1, 1, 1);
		float4 Ks = CalculateSpecularCoefficient(hitPosition, incidentLightRay, normal, specularPower);
		specularColor = specularCoef * Ks * lightSpecularColor;
	}

	// Ambient component.
	// Fake AO: Darken faces with normal facing downwards/away from the sky a little bit.
	float4 ambientColor = g_sceneCB.lightAmbientColor;
	float4 ambientColorMin = g_sceneCB.lightAmbientColor - 0.1;
	float4 ambientColorMax = g_sceneCB.lightAmbientColor;
	float a = 1 - saturate(dot(normal, float3(0, -1, 0)));
	ambientColor = albedo * lerp(ambientColorMin, ambientColorMax, a);

	return ambientColor + diffuseColor + specularColor;
}

// Load three 16 bit indices from a byte addressed buffer.
//uint3 Load3x16BitIndices(uint offsetBytes)
//{
//    uint3 indices;
//
//    // ByteAdressBuffer loads must be aligned at a 4 byte boundary.
//    // Since we need to read three 16 bit indices: { 0, 1, 2 } 
//    // aligned at a 4 byte boundary as: { 0 1 } { 2 0 } { 1 2 } { 0 1 } ...
//    // we will load 8 bytes (~ 4 indices { a b | c d }) to handle two possible index triplet layouts,
//    // based on first index's offsetBytes being aligned at the 4 byte boundary or not:
//    //  Aligned:     { 0 1 | 2 - }
//    //  Not aligned: { - 0 | 1 2 }
//    const uint dwordAlignedOffset = offsetBytes & ~3;    
//    const uint2 four16BitIndices = Indices.Load2(dwordAlignedOffset);//Indices.Load2(dwordAlignedOffset);
// 
//    // Aligned: { 0 1 | 2 - } => retrieve first three 16bit indices
//    if (dwordAlignedOffset == offsetBytes)
//    {
//        indices.x = four16BitIndices.x & 0xffff;
//        indices.y = (four16BitIndices.x >> 16) & 0xffff;
//        indices.z = four16BitIndices.y & 0xffff;
//    }
//    else // Not aligned: { - 0 | 1 2 } => retrieve last three 16bit indices
//    {
//        indices.x = (four16BitIndices.x >> 16) & 0xffff;
//        indices.y = four16BitIndices.y & 0xffff;
//        indices.z = (four16BitIndices.y >> 16) & 0xffff;
//    }
//
//    return indices;
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
//****************------ Shading functions -------***************************
//***************************************************************************

float G2(in float3 l, in float3 v, in float3 n) {
	float3 h = (l - v) / length(l - v);
	float res = min(2 * saturate(dot(n, h))*saturate(dot(-v, n)) / saturate(dot(-v, h)),
		2 * saturate(dot(n, h))*saturate(dot(l, n)) / saturate(dot(-v, h)));
	return min(res, 1);
}

float G1(in float3 v, in float3 n, in float roughness) {
	float k = pow(roughness + 1, 2) / 8;
	return saturate(dot(v, n)) / (saturate(dot(v, n))*(1 - k) + k);
}

float Gue4(in float3 l, in float3 v, in float3 n, in float roughness) {
	return G1(l, n, roughness) * G1(-v, n, roughness);
}

float3 Dgtr(in float3 l, in float3 v, in float3 n, in float3 albedo, in float roughness) {
	float3 h = (l - v) / length(l - v);
	float CosThetaH = saturate(dot(n, h)) / (length(n)*length(h));
	float CosThetaH_Pow2 = pow(CosThetaH, 2);
	return albedo / pow((pow(roughness, 2) * CosThetaH_Pow2 + 1 - CosThetaH_Pow2), 2);
}


float3 DisneyDiffuse(in float3 l, in float3 v, in float3 n, in float3 albedo, in float roughness)
{
	if (dot(l, n) < 0 || dot(-v, n) < 0) return float3(0, 0, 0);
	float3 h = (l - v) / length(l - v);
	float CosThetaL = saturate(dot(l, n)) / (length(l)*length(n));
	float CosThetaV = saturate(dot(-v, n)) / (length(v)*length(n));
	float CosThetaD = saturate(dot(l, h)) / (length(l)*length(h));

	float Fd90 = 0.5 + 2 * roughness * pow(CosThetaD, 2);
	return (albedo / 3.14159) *
		(1 + (Fd90 - 1) * pow(1 - CosThetaL, 5)) *
		(1 + (Fd90 - 1) * pow(1 - CosThetaV, 5));
}

// Fresnel reflectance - schlick approximation.
float3 FresnelReflectanceSchlick(in float3 v, in float3 n, in float3 f0)
{
	float cosi = saturate(dot(-v, n));
	return f0 + (1 - f0)*pow(1 - cosi, 5);
}

//Cook-Torrance BRDF
float3 Cook_Torrance(in float3 l, in float3 v, in float3 n, in float3 albedo, in float roughness) {
	float3 temp = FresnelReflectanceSchlick(v, n, albedo)* Gue4(l, v, n, roughness) * Dgtr(l, v, n, albedo, roughness);
	//; G2(l, v, n)// *Dgtr(l, v, n, albedo, roughness);
	return temp / (3.14 * saturate(dot(n, l)) * saturate(dot(n, -v)));
}


//***************************************************************************
//*****------ TraceRay wrappers for radiance and shadow rays. -------********
//***************************************************************************
// Trace a radiance ray into the scene and returns a shaded color.
float4 TraceRadianceRay(in Ray ray, in UINT currentRayRecursionDepth)
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
	RayPayload rayPayload = { float4(0, 0, 0, 0), currentRayRecursionDepth + 1 };
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
//********************------ Ray gen shader.. -------************************
//***************************************************************************

// Generate a ray in world space for a camera pixel corresponding to an index from the dispatched 2D grid.
inline void GenerateCameraRay(uint2 index, out float3 origin, out float3 direction)
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
    
    // Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
    GenerateCameraRay(DispatchRaysIndex().xy, origin, rayDir);

    // Trace the ray.
    // Set the ray's extents.
    Ray ray;
    ray.origin = origin;
    ray.direction = rayDir;
	float4 color = TraceRadianceRay(ray, 0);
	//TraceRay(Scene, RAY_FLAG_NONE, ~0, 0, 1, 0, ray, payload);//test


    // Write the raytraced color to the output texture.
    RenderTarget[DispatchRaysIndex().xy] = color;
}

[shader("closesthit")]
void MyClosestHitShader(inout RayPayload payload, in MyAttributes attr)
{
	ShadingData hit = GetShadingData(attr);

	float4 fresnelR = float4(FresnelReflectanceSchlick(WorldRayDirection(), hit.normal, g_cubeCB.albedo.xyz), 1);

	float3 hitPosition = HitWorldPosition() + hit.normal * 0.05;
	float3 Hit2Light = normalize(g_sceneCB.lightPosition.xyz - hitPosition);
	Ray shadowRay = { hitPosition,  Hit2Light };
	bool shadowRayHit = TraceShadowRayAndReportIfHit(shadowRay, payload.recursionDepth);

	// Reflected component.
	float4 reflectedColor = float4(0, 0, 0, 0);
	//if (l_materialCB.reflectanceCoef > 0.001)
	{
		//Trace a reflection ray.
		Ray reflectionRay = { hitPosition, reflect(WorldRayDirection(), hit.normal) };
		float4 reflectionColor = TraceRadianceRay(reflectionRay, payload.recursionDepth);
		reflectedColor = 0.3 * fresnelR * reflectionColor;
	}

	float4 phongColor = CalculatePhongLighting(g_cubeCB.albedo, hit.normal, shadowRayHit, g_cubeCB.diffuseCoef, g_cubeCB.specularCoef, g_cubeCB.specularPower);
	
	float4 color = phongColor + reflectedColor;// +reflectedColor;

	float4 disneyDiColor = float4(DisneyDiffuse(Hit2Light, normalize(WorldRayDirection()), hit.normal, g_cubeCB.albedo.xyz, g_cubeCB.roughness), 1);
	
	float4 CookTorranceColor = float4(0, 0, 0, 0);
	if(!shadowRayHit)
	CookTorranceColor = float4(Cook_Torrance(Hit2Light, normalize(WorldRayDirection()), hit.normal, g_cubeCB.albedo.xyz, g_cubeCB.roughness), 1);
	// Apply visibility falloff.
	//float t = RayTCurrent();
	//color = lerp(color, BackgroundColor, 1.0 - exp(-0.000002*t*t*t));

	//(1- fresnelR)* disneyDiColor + fresnelR * CookTorranceColor + reflectedColor
	payload.color = (1 - fresnelR)* disneyDiColor + fresnelR * CookTorranceColor + reflectedColor;// CookTorranceColor;//CalculateDiffuseLighting(HitWorldPosition(), hit.normal); //);//color;
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