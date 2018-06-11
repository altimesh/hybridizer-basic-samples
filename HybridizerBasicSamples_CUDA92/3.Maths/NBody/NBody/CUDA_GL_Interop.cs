using Hybridizer.Runtime.CUDAImports;
using System;
using System.Runtime.InteropServices;

namespace NBody
{
    public class CUDA_GL_Interop
    {
        [DllImport("cudart64_92.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError_t cudaGLRegisterBufferObject(uint buffer);

        [DllImport("cudart64_92.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError_t cudaGraphicsGLRegisterBuffer(out IntPtr pCudaResource, uint buffer, uint Flags);

        [DllImport("cudart64_92.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError_t cudaGraphicsUnregisterResource(IntPtr resource);

        [DllImport("cudart64_92.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError_t cudaGLUnregisterBufferObject(uint buffer);

        [DllImport("cudart64_92.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError_t cudaGraphicsResourceGetMappedPointer(out IntPtr devPtr, out size_t size, IntPtr resource);

        [DllImport("cudart64_92.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError_t cudaGraphicsResourceSetMapFlags(IntPtr resource, uint flags);

        [DllImport("cudart64_92.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError_t cudaGraphicsMapResources(int count, IntPtr[] resources, cudaStream_t stream);

        [DllImport("cudart64_92.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError_t cudaGraphicsUnmapResources(int count, IntPtr[] resources, cudaStream_t stream);
    }
}
