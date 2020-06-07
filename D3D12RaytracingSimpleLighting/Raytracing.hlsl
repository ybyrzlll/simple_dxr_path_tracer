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

void samplingBRDF(out float3 sampleDir, out float sampleProb, out float4 brdfCos,
	in float3 surfaceNormal, in float3 baseDir, in Material mtl, inout uint seed)
{

	float4 brdfEval;
	float4 albedo = mtl.color_diffuse;
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

	//Metal
	//if (mtl.metallic >0.8)
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
	//		//float D = Dgtr(mtl.color_specular, HN*HN, alpha2);
	//		//float G = Gue4(IN, dot(O, N), alpha2);
	//		float D = TrowbridgeReitzGGX(HN*HN, alpha2);
	//		float G = Smith_TrowbridgeReitz(I, O, H, N, alpha2);
	//		float4 F = albedo + (1 - albedo) * pow(max(0, 1 - OH), 5);
	//		brdfEval = ((D * G) / (4 * IN * ON)) * F;
	//		sampleProb = D * HN / (4 * OH);		// IN > 0 imply OH > 0
	//	}
	//}
	//Plastic
	//else if (mtl.metallic < 0.8)
	{
		/*float metallic;
		float specular;*/
		float r = mtl.specular;

		//if (rnd(seed) < r)
		{
			H = sample_hemisphere_TrowbridgeReitzCos(alpha2, seed);
			HN = H.z;
			H = applyRotationMappingZToN(N, H);
			OH = dot(O, H);

			I = 2 * OH * H - O;
			IN = dot(I, N);
		}
		/*else
		{
			I = sample_hemisphere_cos(seed);
			IN = I.z;
			I = applyRotationMappingZToN(N, I);

			H = O + I;
			H = (1 / length(H)) * H;
			HN = dot(H, N);
			OH = dot(O, H);
		}*/

		if (IN < 0)
		{
			brdfEval = 0;
			sampleProb = 0;		//sampleProb = r * (D*HN / (4*abs(OH)));  if allowing sample negative hemisphere
		}
		else
		{
			float D = TrowbridgeReitzGGX(HN*HN, alpha2);
			float G = Smith_TrowbridgeReitz(I, O, H, N, alpha2);

			float4 albedo_dielectric =  float4(0.08f, 0.08f, 0.08f, 1.0f) * mtl.specular *50;

			float metallic = mtl.metallic;
			float4 Rf = (1 - metallic) * albedo_dielectric + metallic * albedo;
			float4 F = Rf + (1 - Rf) * pow(max(0, 1 - OH), 5);
			brdfEval = ((D * G) / (4 * IN * ON)) * F;
			sampleProb = D * HN / (4 * OH);

			/*float4 F = albedo + (1 - albedo) * pow(max(0, 1 - OH), 5);
			brdfEval = ((D * G) / (4 * IN * ON)) * F;
			sampleProb = D * HN / (4 * OH);*/

			/*float3 spec = ((D * G) / (4 * IN * ON));
			brdfEval = (r * float4(spec,1) + (1 - r) * InvPi * albedo, 1);
			sampleProb = r * (D*HN / (4 * OH)) + (1 - r) * (InvPi * IN);*/
		}
	}

	float4 diffuse_color = mtl.color_diffuse * (1 - mtl.metallic);

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
// Trace a radiance ray into the scene and returns a shaded color.
float4 TraceRadianceRay(in Ray ray, in UINT currentRayRecursionDepth, in UINT seed, in float4 attenuation)
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
	RayPayload rayPayload = { currentRayRecursionDepth + 1 ,attenuation, float4(0, 0, 0, 0),  seed};
	TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, rayDesc, rayPayload);

	return rayPayload.radiance;
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
float4 CalculateDiffuseLighting(float3 hitPosition, float3 normal, float4 albedo)
{
    float3 pixelToLight = normalize(g_sceneCB.lightPosition.xyz - hitPosition);

    // Diffuse contribution.
    float fNDotL = max(0.0f, dot(pixelToLight, normal));

    return albedo * g_sceneCB.lightDiffuseColor * fNDotL;
}

[shader("raygeneration")]
void MyRaygenShader()
{
    float3 rayDir;
    float3 origin;
    
	float4 color = { 0,0,0,0 };

	uint2 launchIdx = DispatchRaysIndex().xy;
	uint2 launchDim = DispatchRaysDimensions().xy;
	uint bufferOffset = launchDim.x * launchIdx.y + launchIdx.x;
	uint seed = getNewSeed(bufferOffset, g_sceneCB.frame_num, 8);

    // Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
	[unroll]
	for (int i = 0; i < Sample_Num; i++) {
		float2 random = float2(rnd(seed), rnd(seed));
		GenerateCameraRay((float2)DispatchRaysIndex().xy + random, origin, rayDir);
		//GenerateCameraRay((float2)DispatchRaysIndex().xy , origin, rayDir);
		Ray ray;
		ray.origin = origin;
		ray.direction = rayDir;
		color += TraceRadianceRay(ray, 0, seed, float4(1.0f, 1.0f, 1.0f, 1.0f));
	}
    // Write the raytraced color to the output texture.
    RenderTarget[DispatchRaysIndex().xy] = color / Sample_Num;
}


[shader("closesthit")]
void MyClosestHitShader(inout RayPayload payload, in MyAttributes attr)
{
	ShadingData hit = GetShadingData(attr);

	float3 N = hit.normal, fN, E = -WorldRayDirection();
	//computeNormal(N, fN, attr);
	float EN = dot(E, N);// , EfN = dot(E, fN);


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
		/*Ray sampleRay = { HitWorldPosition(), WorldRayDirection() };
		payload.radiance =  TraceRadianceRay(sampleRay, payload.recursionDepth, payload.seed, payload.attenuation);*/
		return;
	}

	if (any(hit.material.emission)) //光源
	{
		payload.radiance += hit.material.emission;
	}

	float3 sampleDir;
	float4 brdfCos;
	float sampleProb;
	samplingBRDF(sampleDir, sampleProb, brdfCos, N, E, hit.material, payload.seed);

	if (dot(sampleDir, N) <= 0) {
		//Stop! ==没有折射
		return; // payload.rayDepth = maxPathLength;
	}
		
	Ray sampleRay = { HitWorldPosition(), sampleDir };
	float4 radiance = payload.attenuation * payload.radiance;// 
	if (!any(hit.material.emission)) //!光源
	{
		float4 attenuation = brdfCos / sampleProb;
		payload.attenuation *= attenuation;
	}

	float4 sampleColor = TraceRadianceRay(sampleRay, payload.recursionDepth, payload.seed, payload.attenuation);
	payload.radiance = radiance +sampleColor;
}

[shader("miss")]
void MyMissShader(inout RayPayload payload)
{
    payload.radiance = BackgroundColor;
}

[shader("miss")]
void MyMissShader_ShadowRay(inout ShadowRayPayload rayPayload)
{
	rayPayload.hit = false;
}

#endif // RAYTRACING_HLSL