#pragma once

#include "cuda_runtime.h"
#include "vector_types.h"
#include "sutil/vec_math.h"

#define _define_math_operator(op) \
	SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator op(const float& a, const float3& b) { return make_float3(a op b.x, a op b.y, a op b.z); } \
	SUTIL_INLINE SUTIL_HOSTDEVICE int3 operator op(const int& a, const int3& b) { return make_int3(a op b.x, a op b.y, a op b.z); }

_define_math_operator(*);
_define_math_operator(+);
_define_math_operator(-);
_define_math_operator(/);
