//***************************************************************************
//****************------ Old Shading functions -------***********************
//***************************************************************************

float G2(in float NoV, in float NoH, in float NoL) {
	float res = min(2 * NoH * NoV / NoH, 2 * NoH * NoL / NoH);
	return min(res, 1);
}

float G1(in float NoV, in float roughness) {
	float k = pow(roughness + 1.0, 2) / 8.0;
	return NoV / (NoV*(1.0 - k) + k);
}

float Gue4(float NoV, float NoL, in float roughness) {
	return G1(NoV, roughness) * G1(NoL, roughness);
}


float3 Dgtr(in float3 SpecularColor, in float NoH, in float roughness) {
	float NoH2 = pow(NoH, 2);
	return SpecularColor / pow((roughness * roughness * NoH2 + 1 - NoH2), 2);
}

float3 DisneyDiffuse(float3 DiffuseColor, float Roughness, float NoV, float NoL, float VoH)
{
	float FD90 = 0.5 + 2.0 * Roughness * VoH * VoH;
	float FdV = 1.0 + (FD90 - 1.0) * pow(1.0 - NoL, 5);
	float FdL = 1.0 + (FD90 - 1.0) * pow(1.0 - NoV, 5);
	return (DiffuseColor / PI) * FdV * FdL;
}

// Fresnel reflectance - schlick approximation.
float3 FresnelReflectanceSchlick(in float NoL, in float3 f0)
{
	return f0 + (1 - f0)*pow(1 - NoL, 5);
}



//***************************************************************************
//****************------ New Shading functions -------***************************
//***************************************************************************
float3 Diffuse_OrenNayar(float3 DiffuseColor, float Roughness, float NoV, float NoL, float VoH)
{
	float a = Roughness * Roughness;
	float s = a;// / ( 1.29 + 0.5 * a );
	float s2 = s * s;
	float VoL = 2 * VoH * VoH - 1;		// double angle identity
	float Cosri = VoL - NoV * NoL;
	float C1 = 1 - 0.5 * s2 / (s2 + 0.33);
	float C2 = 0.45 * s2 / (s2 + 0.09) * Cosri * (Cosri >= 0 ? rcp(max(NoL, NoV)) : 1);
	return DiffuseColor / PI * (C1 + C2) * (1 + Roughness * 0.5);
}

float Pow5(float x)
{
	float xx = x * x;
	return xx * xx * x;
}

float3 Diffuse_Burley(float3 DiffuseColor, float Roughness, float NoV, float NoL, float VoH)
{
	float FD90 = 0.5 + 2 * VoH * VoH * Roughness;
	float FdV = 1 + (FD90 - 1) * Pow5(1 - NoV);
	float FdL = 1 + (FD90 - 1) * Pow5(1 - NoL);
	return DiffuseColor * ((1 / PI) * FdV * FdL);
}

// Fresnel reflectance - schlick approximation.
float3 F_Schlick(in float3 SpecularColor, in float VoH)
{
	float Fc = Pow5(1 - VoH);					// 1 sub, 3 mul
	//return Fc + (1 - Fc) * SpecularColor;		// 1 add, 3 mad

	// Anything less than 2% is physically impossible and is instead considered to be shadowing
	return saturate(50.0 * SpecularColor.g) * Fc + (1 - Fc) * SpecularColor;
}

// Fresnel reflectance - schlick approximation.
float3 F_Schlick2(in float3 SpecularColor, in float3 V, in float3 L)
{
	float VoL = dot(V, L);
	float InvLenH = rsqrt(2 + 2 * VoL);
	float VoH = saturate(InvLenH + InvLenH * VoL);

	float Fc = Pow5(1 - VoH);					// 1 sub, 3 mul
	//return Fc + (1 - Fc) * SpecularColor;		// 1 add, 3 mad

	// Anything less than 2% is physically impossible and is instead considered to be shadowing
	return saturate(50.0 * SpecularColor.g) * Fc + (1 - Fc) * SpecularColor;
}

float Vis_Schlick(float a2, float NoV, float NoL)
{
	float k = sqrt(a2) * 0.5;
	float Vis_SchlickV = NoV * (1 - k) + k;
	float Vis_SchlickL = NoL * (1 - k) + k;
	return 0.25 / (Vis_SchlickV * Vis_SchlickL);
}

float3 DGGX(in float NoH, in float Roughness)
{
	float a = Roughness * Roughness;
	float a2 = a * a;
	float d = (NoH * a2 - NoH) * NoH + 1;	// 2 mad  ??Warning
	return a2 / (PI*d*d);
}

float3 Cook_Torrance(in float3 SpecularColor, in float Roughness, in float NoV, in float NoL, in float VoH, in float NoH) {
	//float3 temp = (F_Schlick(SpecularColor, VoH))* saturate(Vis_Schlick(Roughness * Roughness, NoV, NoL)) * saturate(DGGX(NoH, Roughness));
	float3 temp = (F_Schlick(SpecularColor, VoH))* (G2(NoV, NoH, NoL)) *(DGGX(NoH, Roughness));//(G2(NoV, NoH, NoL)); 
	//float3 temp = (F_Schlick(SpecularColor, VoH)) * (G2(NoV, NoH, NoL)) * Dgtr(SpecularColor, NoH, Roughness);
	float temp2 = clamp(PI * NoL * NoV, 0.01, 0.99);
	return temp / (PI * NoL * NoV);
}

//Cook-Torrance BRDF
float3 Cook_Torrance2(in float3 SpecularColor, in float Roughness, in float NoV, in float NoL, in float VoH, in float NoH) {
	//float3 temp = FresnelReflectanceSchlick(NoL, SpecularColor)* Gue4(NoV, NoL, Roughness) * Dgtr(SpecularColor, NoH, Roughness);//* DGGX(NoH, Roughness)
	float3 temp = F_Schlick(SpecularColor, VoH) * saturate(Gue4(NoV, NoL, Roughness)) * DGGX(NoH, Roughness);//* DGGX(NoH, Roughness)
	return temp / (PI * NoL * NoV);
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
	return float4(Cook_Torrance2(SpecColor, Roughness, NoV, NoL, VoH, NoH), 1.0);
}

//***************************************************************************
//********************------ Helper -------*************************
//***************************************************************************
//float GetSpecularEventProbability(float3 SpecularColor, float NoV)
//{
//	float3 F = F_Schlick(SpecularColor, NoV);
//	return (F.x + F.y + F.z) / 3.f;
//}