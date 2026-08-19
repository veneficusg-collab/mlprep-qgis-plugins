#ifndef STEP4_GDAL_COMPAT_OGR_SPATIALREF_H
#define STEP4_GDAL_COMPAT_OGR_SPATIALREF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* OGRSpatialReferenceH;
typedef void* OGRDataSourceH;

OGRSpatialReferenceH OSRNewSpatialReference(const char*);
int OSRIsGeographic(OGRSpatialReferenceH);
double OSRGetLinearUnits(OGRSpatialReferenceH, char**);

#ifdef __cplusplus
}
#endif

#endif
