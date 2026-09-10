#ifndef STEP4_GDAL_COMPAT_CPL_CONV_H
#define STEP4_GDAL_COMPAT_CPL_CONV_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void* CPLMalloc(size_t);
void CPLFree(void*);

#ifdef __cplusplus
}
#endif

#endif
