using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace TextureAndSurface
{
    public class TextureHelpers
    {

        [IntrinsicFunction("surf2Dwrite")]
        public static void surf2Dwrite(float data, cudaSurfaceObject_t surfObj, int x, int y)
        {
        }

        [IntrinsicFunction("tex2D<float>")]
        public static float tex2D(cudaTextureObject_t texObj, float x, float y)
        {
            return 0.0F;
        }

        [IntrinsicFunction("cudaCreateChannelDesc")]
        public static cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind kind)
        {
            cudaChannelFormatDesc res;
            res.x = x;
            res.y = y;
            res.z = z;
            res.w = w;
            res.f = kind;
            return res;
        }

        public static unsafe cudaResourceDesc CreateCudaResourceDesc(cudaArray_t cuArrayTex)
        {
            cudaResourceDesc resDescTex;
            memset((byte*)&resDescTex, Marshal.SizeOf(resDescTex), 0);
            resDescTex.arrayStruct = cuArrayTex;
            resDescTex.resType = cudaResourceType.cudaResourceTypeArray;
            return resDescTex;
        }

        public static cudaArray_t CreateCudaArray(cudaChannelFormatDesc channelDescTex, IntPtr src, int width, int height)
        {
            cudaArray_t cuArrayTex;
            cuda.MallocArray(out cuArrayTex, ref channelDescTex, width, height, cudaMallocArrayFlags.cudaArrayDefault);
            cuda.MemcpyToArray(cuArrayTex, 0, 0, src, width * height * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);
            return cuArrayTex;
        }
        public static unsafe cudaTextureDesc CreateCudaTextureDesc()
        {
            cudaTextureDesc texDesc;
            memset((byte*)&texDesc, Marshal.SizeOf(texDesc), 0);
            texDesc.addressMode[0] = (int)cudaTextureAddressMode.cudaAddressModeWrap;
            texDesc.addressMode[1] = (int)cudaTextureAddressMode.cudaAddressModeWrap;
            texDesc.filterMode = cudaTextureFilterMode.cudaFilterModePoint;
            texDesc.readMode = cudaTextureReadMode.cudaReadModeElementType;
            texDesc.normalizedCoords = 0;
            return texDesc;
        }

        public static unsafe void memset(byte* data, int bytes, byte val)
        {
            for (int i = 0; i < bytes; ++i)
            {
                data[i] = val;
            }
        }
    }
}