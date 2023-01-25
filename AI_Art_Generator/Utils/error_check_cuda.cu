// Standard Library
#include <iostream>

#include <driver_types.h>

// File header
#include "AI_Art_Generator/Utils/error_check_cuda.hpp"

static std::string cuda_return_error_string(cudaError_t error)
{
    switch (error)
    {
    case cudaErrorInvalidValue:                   return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation:               return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError:            return "cudaErrorInitializationError";
    case cudaErrorCudartUnloading:                return "cudaErrorCudartUnloading";
    case cudaErrorProfilerDisabled:               return "cudaErrorProfilerDisabled";
    case cudaErrorProfilerNotInitialized:         return "cudaErrorProfilerNotInitialized";
    case cudaErrorProfilerAlreadyStarted:         return "cudaErrorProfilerAlreadyStarted";
    case cudaErrorProfilerAlreadyStopped:         return "cudaErrorProfilerAlreadyStopped";
    case cudaErrorInvalidConfiguration:           return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidPitchValue:              return "cudaErrorInvalidPitchValue";
    case cudaErrorInvalidSymbol:                  return "cudaErrorInvalidSymbol";
    case cudaErrorInvalidHostPointer:             return "cudaErrorInvalidHostPointer";
    case cudaErrorInvalidDevicePointer:           return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidTexture:                 return "cudaErrorInvalidTexture";
    case cudaErrorInvalidTextureBinding:          return "cudaErrorInvalidTextureBinding";
    case cudaErrorInvalidChannelDescriptor:       return "cudaErrorInvalidChannelDescriptor";
    case cudaErrorInvalidMemcpyDirection:         return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorAddressOfConstant:              return "cudaErrorAddressOfConstant";
    case cudaErrorTextureFetchFailed:             return "cudaErrorTextureFetchFailed";
    case cudaErrorTextureNotBound:                return "cudaErrorTextureNotBound";
    case cudaErrorSynchronizationError:           return "cudaErrorSynchronizationError";
    case cudaErrorInvalidFilterSetting:           return "cudaErrorInvalidFilterSetting";
    case cudaErrorInvalidNormSetting:             return "cudaErrorInvalidNormSetting";
    case cudaErrorMixedDeviceExecution:           return "cudaErrorMixedDeviceExecution";
    case cudaErrorNotYetImplemented:              return "cudaErrorNotYetImplemented";
    case cudaErrorMemoryValueTooLarge:            return "cudaErrorMemoryValueTooLarge";
    case cudaErrorStubLibrary:                    return "cudaErrorStubLibrary";
    case cudaErrorInsufficientDriver:             return "cudaErrorInsufficientDriver";
    case cudaErrorCallRequiresNewerDriver:        return "cudaErrorCallRequiresNewerDriver";
    case cudaErrorInvalidSurface:                 return "cudaErrorInvalidSurface";
    case cudaErrorDuplicateVariableName:          return "cudaErrorDuplicateVariableName";
    case cudaErrorDuplicateTextureName:           return "cudaErrorDuplicateTextureName";
    case cudaErrorDuplicateSurfaceName:           return "cudaErrorDuplicateSurfaceName";
    case cudaErrorDevicesUnavailable:             return "cudaErrorDevicesUnavailable";
    case cudaErrorIncompatibleDriverContext:      return "cudaErrorIncompatibleDriverContext";
    case cudaErrorMissingConfiguration:           return "cudaErrorMissingConfiguration";
    case cudaErrorPriorLaunchFailure:             return "cudaErrorPriorLaunchFailure";
    case cudaErrorLaunchMaxDepthExceeded:         return "cudaErrorLaunchMaxDepthExceeded";
    case cudaErrorLaunchFileScopedTex:            return "cudaErrorLaunchFileScopedTex";
    case cudaErrorLaunchFileScopedSurf:           return "cudaErrorLaunchFileScopedSurf";
    case cudaErrorSyncDepthExceeded:              return "cudaErrorSyncDepthExceeded";
    case cudaErrorLaunchPendingCountExceeded:     return "cudaErrorLaunchPendingCountExceeded";
    case cudaErrorInvalidDeviceFunction:          return "cudaErrorInvalidDeviceFunction";
    case cudaErrorNoDevice:                       return "cudaErrorNoDevice";
    case cudaErrorInvalidDevice:                  return "cudaErrorInvalidDevice";
    case cudaErrorDeviceNotLicensed:              return "cudaErrorDeviceNotLicensed";
    case cudaErrorSoftwareValidityNotEstablished: return "cudaErrorSoftwareValidityNotEstablished";
    case cudaErrorStartupFailure:                 return "cudaErrorStartupFailure";
    case cudaErrorInvalidKernelImage:             return "cudaErrorInvalidKernelImage";
    case cudaErrorDeviceUninitialized:            return "cudaErrorDeviceUninitialized";
    case cudaErrorMapBufferObjectFailed:          return "cudaErrorMapBufferObjectFailed";
    case cudaErrorUnmapBufferObjectFailed:        return "cudaErrorUnmapBufferObjectFailed";
    case cudaErrorArrayIsMapped:                  return "cudaErrorArrayIsMapped";
    case cudaErrorAlreadyMapped:                  return "cudaErrorAlreadyMapped";
    case cudaErrorNoKernelImageForDevice:         return "cudaErrorNoKernelImageForDevice";
    case cudaErrorAlreadyAcquired:                return "cudaErrorAlreadyAcquired";
    case cudaErrorNotMapped:                      return "cudaErrorNotMapped";
    case cudaErrorNotMappedAsArray:               return "cudaErrorNotMappedAsArray";
    case cudaErrorNotMappedAsPointer:             return "cudaErrorNotMappedAsPointer";
    case cudaErrorECCUncorrectable:               return "cudaErrorECCUncorrectable";
    case cudaErrorUnsupportedLimit:               return "cudaErrorUnsupportedLimit";
    case cudaErrorDeviceAlreadyInUse:             return "cudaErrorDeviceAlreadyInUse";
    case cudaErrorPeerAccessUnsupported:          return "cudaErrorPeerAccessUnsupported";
    case cudaErrorInvalidPtx:                     return "cudaErrorInvalidPtx";
    case cudaErrorInvalidGraphicsContext:         return "cudaErrorInvalidGraphicsContext";
    case cudaErrorNvlinkUncorrectable:            return "cudaErrorNvlinkUncorrectable";
    case cudaErrorJitCompilerNotFound:            return "cudaErrorJitCompilerNotFound";
    case cudaErrorUnsupportedPtxVersion:          return "cudaErrorUnsupportedPtxVersion";
    case cudaErrorJitCompilationDisabled:         return "cudaErrorJitCompilationDisabled";
    case cudaErrorUnsupportedExecAffinity:        return "cudaErrorUnsupportedExecAffinity";
    case cudaErrorInvalidSource:                  return "cudaErrorInvalidSource";
    case cudaErrorFileNotFound:                   return "cudaErrorFileNotFound";
    case cudaErrorSharedObjectSymbolNotFound:     return "cudaErrorSharedObjectSymbolNotFound";
    case cudaErrorSharedObjectInitFailed:         return "cudaErrorSharedObjectInitFailed";
    case cudaErrorOperatingSystem:                return "cudaErrorOperatingSystem";
    case cudaErrorInvalidResourceHandle:          return "cudaErrorInvalidResourceHandle";
    case cudaErrorIllegalState:                   return "cudaErrorIllegalState";
    case cudaErrorSymbolNotFound:                 return "cudaErrorSymbolNotFound";
    case cudaErrorNotReady:                       return "cudaErrorNotReady";
    case cudaErrorIllegalAddress:                 return "cudaErrorIllegalAddress";
    case cudaErrorLaunchOutOfResources:           return "cudaErrorLaunchOutOfResources";
    case cudaErrorLaunchTimeout:                  return "cudaErrorLaunchTimeout";
    case cudaErrorLaunchIncompatibleTexturing:    return "cudaErrorLaunchIncompatibleTexturing";
    case cudaErrorPeerAccessAlreadyEnabled:       return "cudaErrorPeerAccessAlreadyEnabled";
    case cudaErrorPeerAccessNotEnabled:           return "cudaErrorPeerAccessNotEnabled";
    case cudaErrorSetOnActiveProcess:             return "cudaErrorSetOnActiveProcess";
    case cudaErrorContextIsDestroyed:             return "cudaErrorContextIsDestroyed";
    case cudaErrorAssert:                         return "cudaErrorAssert";
    case cudaErrorTooManyPeers:                   return "cudaErrorTooManyPeers";
    case cudaErrorHostMemoryAlreadyRegistered:    return "cudaErrorHostMemoryAlreadyRegistered";
    case cudaErrorHostMemoryNotRegistered:        return "cudaErrorHostMemoryNotRegistered";
    case cudaErrorHardwareStackError:             return "cudaErrorHardwareStackError";
    case cudaErrorIllegalInstruction:             return "cudaErrorIllegalInstruction";
    case cudaErrorMisalignedAddress:              return "cudaErrorMisalignedAddress";
    case cudaErrorInvalidAddressSpace:            return "cudaErrorInvalidAddressSpace";
    case cudaErrorInvalidPc:                      return "cudaErrorInvalidPc";
    case cudaErrorLaunchFailure:                  return "cudaErrorLaunchFailure";
    case cudaErrorCooperativeLaunchTooLarge:      return "cudaErrorCooperativeLaunchTooLarge";
    case cudaErrorNotPermitted:                   return "cudaErrorNotPermitted";
    case cudaErrorNotSupported:                   return "cudaErrorNotSupported";
    case cudaErrorSystemNotReady:                 return "cudaErrorSystemNotReady";
    case cudaErrorSystemDriverMismatch:           return "cudaErrorSystemDriverMismatch";
    case cudaErrorCompatNotSupportedOnDevice:     return "cudaErrorCompatNotSupportedOnDevice";
    case cudaErrorMpsConnectionFailed:            return "cudaErrorMpsConnectionFailed";
    case cudaErrorMpsRpcFailure:                  return "cudaErrorMpsRpcFailure";
    case cudaErrorMpsServerNotReady:              return "cudaErrorMpsServerNotReady";
    case cudaErrorMpsMaxClientsReached:           return "cudaErrorMpsMaxClientsReached";
    case cudaErrorMpsMaxConnectionsReached:       return "cudaErrorMpsMaxConnectionsReached";
    case cudaErrorMpsClientTerminated:            return "cudaErrorMpsClientTerminated";
    case cudaErrorCdpNotSupported:                return "cudaErrorCdpNotSupported";
    case cudaErrorCdpVersionMismatch:             return "cudaErrorCdpVersionMismatch";
    case cudaErrorStreamCaptureUnsupported:       return "cudaErrorStreamCaptureUnsupported";
    case cudaErrorStreamCaptureInvalidated:       return "cudaErrorStreamCaptureInvalidated";
    case cudaErrorStreamCaptureMerge:             return "cudaErrorStreamCaptureMerge";
    case cudaErrorStreamCaptureUnmatched:         return "cudaErrorStreamCaptureUnmatched";
    case cudaErrorStreamCaptureUnjoined:          return "cudaErrorStreamCaptureUnjoined";
    case cudaErrorStreamCaptureIsolation:         return "cudaErrorStreamCaptureIsolation";
    case cudaErrorStreamCaptureImplicit:          return "cudaErrorStreamCaptureImplicit";
    case cudaErrorCapturedEvent:                  return "cudaErrorCapturedEvent";
    case cudaErrorStreamCaptureWrongThread:       return "cudaErrorStreamCaptureWrongThread";
    case cudaErrorTimeout:                        return "cudaErrorTimeout";
    case cudaErrorGraphExecUpdateFailure:         return "cudaErrorGraphExecUpdateFailure";
    case cudaErrorExternalDevice:                 return "cudaErrorExternalDevice";
    case cudaErrorInvalidClusterSize:             return "cudaErrorInvalidClusterSize";
    case cudaErrorUnknown:                        return "cudaErrorUnknown";
    case cudaErrorApiFailureBase:                 return "cudaErrorApiFailureBase";
    }

    return "<unknown>";
}

void cuda_check_expanded(cudaError_t result, const char *function_name, const char *filename, int line_number)
{
    if(result)
    {
        std::cout << "CUDA Error: " << cuda_return_error_string(result) << " (" << result << "), Function name: " << function_name << ", Filename:" << filename << ":" << line_number << std::endl;
    }

    return;
}
