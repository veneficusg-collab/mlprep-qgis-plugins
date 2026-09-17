#ifndef STEP4_GDAL_COMPAT_GDAL_H
#define STEP4_GDAL_COMPAT_GDAL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* GDALDatasetH;
typedef void* GDALRasterBandH;
typedef void* GDALDriverH;

typedef enum {
    GA_ReadOnly = 0,
    GA_Update = 1
} GDALAccess;

typedef enum {
    GF_Read = 0,
    GF_Write = 1
} GDALRWFlag;

typedef enum {
    GDT_Unknown = 0,
    GDT_Byte = 1,
    GDT_UInt16 = 2,
    GDT_Int16 = 3,
    GDT_UInt32 = 4,
    GDT_Int32 = 5,
    GDT_Float32 = 6,
    GDT_Float64 = 7,
    GDT_CInt16 = 8,
    GDT_CInt32 = 9,
    GDT_CFloat32 = 10,
    GDT_CFloat64 = 11,
    GDT_TypeCount = 12
} GDALDataType;

typedef enum {
    CE_None = 0,
    CE_Debug = 1,
    CE_Warning = 2,
    CE_Failure = 3,
    CE_Fatal = 4
} CPLErr;

#ifndef FALSE
#define FALSE 0
#endif

void GDALAllRegister(void);
GDALDatasetH GDALOpen(const char*, GDALAccess);
void GDALClose(GDALDatasetH);
GDALDriverH GDALGetDatasetDriver(GDALDatasetH);
GDALDriverH GDALGetDriverByName(const char*);
GDALDatasetH GDALCreate(GDALDriverH, const char*, int, int, int, GDALDataType, char**);
GDALRasterBandH GDALGetRasterBand(GDALDatasetH, int);
int GDALGetRasterXSize(GDALDatasetH);
int GDALGetRasterYSize(GDALDatasetH);
const char* GDALGetProjectionRef(GDALDatasetH);
CPLErr GDALSetProjection(GDALDatasetH, const char*);
CPLErr GDALGetGeoTransform(GDALDatasetH, double*);
CPLErr GDALSetGeoTransform(GDALDatasetH, double*);
double GDALGetRasterNoDataValue(GDALRasterBandH, int*);
CPLErr GDALSetRasterNoDataValue(GDALRasterBandH, double);
const char* GDALGetRasterUnitType(GDALRasterBandH);
GDALDataType GDALGetRasterDataType(GDALRasterBandH);
CPLErr GDALRasterIO(
    GDALRasterBandH,
    GDALRWFlag,
    int,
    int,
    int,
    int,
    void*,
    int,
    int,
    GDALDataType,
    int,
    int);
void GDALFlushCache(GDALDatasetH);

#ifdef __cplusplus
}
#endif

#endif
