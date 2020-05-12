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

RaytracingAccelerationStructure Scene : register(t0, space0);
RWTexture2D<float4> RenderTarget : register(u0);
StructuredBuffer<Mesh> scene_meshes : register(t1);//HLSL_REGISTER_MESHES
StructuredBuffer<Vertex> Vertices : register(t2);//HLSL_REGISTER_VERTICES
StructuredBuffer<uint> Indices : register(t3); //HLSL_REGISTER_INDICES
StructuredBuffer<Material> scene_materials : register(t4);//HLSL_REGISTER_MATERIALS

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
	Material material;
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
	data.material = scene_materials[scene_meshes[InstanceID()].material];//

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

	uint screen_seed = initRand(g_sceneCB.frame_num, g_sceneCB.frame_num+1, 16);

    // Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
	[unroll]
	for (int i = 0; i < Sample_Num; i++) {
		/*float2 random = float2(nextRand(screen_seed), nextRand(screen_seed));
		GenerateCameraRay((float2)DispatchRaysIndex().xy + random, origin, rayDir);*/
		GenerateCameraRay((float2)DispatchRaysIndex().xy , origin, rayDir);
		Ray ray;
		ray.origin = origin;
		ray.direction = rayDir;
		color += TraceRadianceRay(ray, 0);
	}
    // Write the raytraced color to the output texture.
    RenderTarget[DispatchRaysIndex().xy] = color / Sample_Num;
}


[shader("closesthit")]
void MyClosestHitShader(inout RayPayload payload, in MyAttributes attr)
{
	ShadingData hit = GetShadingData(attr);
	if (hit.material.emission == 1) 
	{
		payload.color = float4(0.1, 1, 0.1, 1.0);
		return;
	}

	float3 hitPosition = HitWorldPosition() + hit.normal * 0.05;
	float3 Hit2Light = normalize(g_sceneCB.lightPosition.xyz - hitPosition);
	Ray shadowRay = { hitPosition,  Hit2Light };
	bool shadowRayHit = TraceShadowRayAndReportIfHit(shadowRay, payload.recursionDepth);
	float3 L = WorldRayDirection();
	float3 V = Hit2Light;
	float3 N = hit.normal;

	float4 fresnelR = float4(F_Schlick2(g_cubeCB.albedo.xyz, V, L), 1.0);

	// Reflected component.
	float4 reflectedColor = float4(0, 0, 0, 0);
	//if (l_materialCB.reflectanceCoef > 0.001)
	{
		//Trace a reflection ray.
		Ray reflectionRay = { hitPosition, reflect(WorldRayDirection(), hit.normal) };
		float4 reflectionColor = TraceRadianceRay(reflectionRay, payload.recursionDepth);
		reflectedColor = 0.3 * fresnelR * reflectionColor;
	}

	//float4 phongColor = CalculatePhongLighting(g_cubeCB.albedo, hit.normal, shadowRayHit, g_cubeCB.diffuseCoef, g_cubeCB.specularCoef, g_cubeCB.specularPower);
	
	float4 ambient = float4(0.1, 0.1, 0.1, 1.0);
	
	

	float4 DiffuseColor = DiffuseShade(g_cubeCB.albedo.xyz, g_cubeCB.roughness, L, V, N);
	
	float4 SpecularColor = float4(0, 0, 0, 0);

	/*if (!shadowRayHit) {
		payload.color += disneyDiColor;
	}*/
	SpecularColor = SpecularShade(g_cubeCB.albedo.xyz, g_cubeCB.roughness, -L, V, N);

	// Apply visibility falloff.
	//float t = RayTCurrent();
	//color = lerp(color, BackgroundColor, 1.0 - exp(-0.000002*t*t*t));

	//(1- fresnelR)* DiffuseColor + fresnelR * CookTorranceColor + reflectedColor
	float4 res = float4(0, 0, 0, 0); 
	//res += saturate(DiffuseColor);// *(1 - fresnelR);//*(1- fresnelR) 
	//res += saturate(SpecularColor);// *fresnelR;
	//res += reflectedColor;
	res += CalculateDiffuseLighting(HitWorldPosition(), hit.normal);
	payload.color = res;// CookTorranceColor;//CalculateDiffuseLighting(HitWorldPosition(), hit.normal); //);//color;
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