#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#include "cuda.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "camera.h"

void render(uchar4* ptr, cudaGraphicsResource *resource, Camera& camera, int width, int height);