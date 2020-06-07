#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers.
#endif

#include <windows.h>

// C RunTime Header Files
#include <stdlib.h>
#include <sstream>
#include <iomanip>

#include <list>
#include <string>
#include <wrl.h>
#include <shellapi.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <atlbase.h>
#include <assert.h>

#include <dxgi1_6.h>
#include "d3d12_1.h"
#include <atlbase.h>
#include "D3D12RaytracingFallback.h"
#include "D3D12RaytracingHelpers.hpp"
#include "d3dx12.h"

#include <DirectXMath.h>

#ifdef _DEBUG
#include <dxgidebug.h>
#endif

#include "DXSampleHelper.h"
#include "DeviceResources.h"

//loadMesh
#include <SDKDDKVer.h>
#define WIN32_LEAN_AND_MEAN // 从 Windows 头中排除极少使用的资料
#include <windows.h>
#include <tchar.h>
#include <wrl.h>  //添加WTL支持 方便使用COM
#include <strsafe.h>
#include <atlcoll.h> //for atl array
#include <atlconv.h> //for T2A
#include <fstream>  //for ifstream

//mine
#include "Buffer.h"
#include "Model.h"
#include "rtrt.h"


using namespace std;
using namespace Microsoft;
using namespace Microsoft::WRL;
using namespace DirectX;

#define RELEASE_SAFE(ptr) { if (ptr != nullptr) { ptr->Release(); ptr = nullptr; } }
#define RELEASE_EXPLICIT(ptr) { if (ptr != nullptr) { ptr->Release(); ptr = nullptr; } else { BREAK("Explicit release on object which has not been initialized."); } }
#define RELEASE RELEASE_SAFE

