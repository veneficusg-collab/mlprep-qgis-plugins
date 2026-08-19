#ifndef STEP4_GDAL_COMPAT_OGR_API_H
#define STEP4_GDAL_COMPAT_OGR_API_H

#include "ogr_spatialref.h"

typedef void* OGRLayerH;
typedef int OGRwkbGeometryType;

#ifdef __cplusplus
extern "C" {
#endif

int OGR_DS_GetLayerCount(OGRDataSourceH);
OGRLayerH OGR_DS_GetLayer(OGRDataSourceH, int);
const char* OGR_L_GetName(OGRLayerH);
OGRwkbGeometryType OGR_L_GetGeomType(OGRLayerH);

#ifdef __cplusplus
}
#endif

#endif
