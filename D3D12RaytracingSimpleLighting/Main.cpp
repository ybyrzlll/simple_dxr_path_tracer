#include "stdafx.h"
#include "D3D12RaytracingSimpleLighting.h"

_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
    D3D12RaytracingSimpleLighting sample(1280, 720, L"D3D12 Raytracing - Simple Lighting");
	//D3D12RaytracingSimpleLighting sample(640, 640, L"D3D12 Raytracing - Simple Lighting");
    return Win32Application::Run(&sample, hInstance, nCmdShow);
}
