#pragma once
#include "RaytracingHlslCompat.h"

namespace rtrt {
	class Model {
	public:
		Model();
		~Model();
		std::vector<std::vector<Vertex>> vertices;
		std::vector<std::vector<Index>> indices;

	};
}
	
