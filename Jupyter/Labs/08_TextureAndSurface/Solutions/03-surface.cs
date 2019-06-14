using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Hybridizer.Runtime.CUDAImports;

namespace TextureAndSurface
{
    class Program
    {
        static void Main(string[] args)
        {
            HybRunner runner = HybRunner.Cuda().SetDistrib(32, 32, 16, 16, 1, 0);
            
            GrayBitmap image = GrayBitmap.Load("../../../images/lena512.bmp");
            uint height = image.Height, width = image.Width;
            ushort[] inputPixels = image.PixelsUShort;
            
            float[] imageFloat = new float[width * height];

            for (int i = 0; i < width * height; ++i)
            {
                imageFloat[i] = (float)inputPixels[i];
            }

            IntPtr src = runner.Marshaller.MarshalManagedToNative(imageFloat);

            //bind texture
            cudaChannelFormatDesc channelDescTex = TextureHelpers.cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat);
            cudaArray_t cuArrayTex = TextureHelpers.CreateCudaArray(channelDescTex, src, (int)width, (int)height);
            cudaResourceDesc resDescTex = TextureHelpers.CreateCudaResourceDesc(cuArrayTex);

            //create Texture descriptor
            cudaTextureDesc texDesc = TextureHelpers.CreateCudaTextureDesc();

            //create Texture object
            cudaTextureObject_t texObj;
            cuda.CreateTextureObject(out texObj, ref resDescTex, ref texDesc);

            //bind surface
            cudaChannelFormatDesc channelDescSurf = TextureHelpers.cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat);
            cudaArray_t cuArraySurf;
            cuda.MallocArray(out cuArraySurf, ref channelDescSurf, width, height, cudaMallocArrayFlags.cudaArraySurfaceLoadStore);

            //create cudaResourceDesc for surface
            cudaResourceDesc resDescSurf = TextureHelpers.CreateCudaResourceDesc(cuArraySurf);

            //create surface object
            cudaSurfaceObject_t surfObj;
            cuda.CreateSurfaceObject(out surfObj, ref resDescSurf);

            dynamic wrapper = runner.Wrap(new Program());

            wrapper.Sobel(texObj, surfObj, (int)width, (int)height);

            //pinned float array to allow the copy of the surface object on the host
            float[] imageCompute = new float[width * height];
            GCHandle handle = GCHandle.Alloc(imageCompute, GCHandleType.Pinned);
            IntPtr dest = handle.AddrOfPinnedObject();

            cuda.MemcpyFromArray(dest, cuArraySurf, 0, 0, width * height * sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost);

            ushort[] outputPixel = new ushort[width * height];
            for (int i = 0; i < width * height; ++i)
            {
                outputPixel[i] = (ushort)imageCompute[i];
            }

            GrayBitmap imageSobel = new GrayBitmap(width, height);
            imageSobel.PixelsUShort = outputPixel;
            imageSobel.Save("../../../output-3-surface/sobel.bmp");
        }

        [EntryPoint]
        public static void Sobel(cudaTextureObject_t texObj, cudaSurfaceObject_t surfObj, int width, int height)
        {
            for (int i = threadIdx.y + blockIdx.y * blockDim.y; i < height; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < width; j += blockDim.x * gridDim.x)
                {
                    int pixelId = i * width + j;
                    if (i != 0 && j != 0 && i != height - 1 && j != width - 1)
                    {
                        float tl = TextureHelpers.tex2D(texObj, j - 1.0F, i - 1.0F);
                        float t = TextureHelpers.tex2D(texObj, j - 1.0F, i);
                        float tr = TextureHelpers.tex2D(texObj, j - 1.0F, i + 1.0F);
                        float l = TextureHelpers.tex2D(texObj, j, i - 1.0F);
                        float r = TextureHelpers.tex2D(texObj, j, i + 1.0F);
                        float br = TextureHelpers.tex2D(texObj, j + 1.0F, i + 1.0F);
                        float bl = TextureHelpers.tex2D(texObj, j + 1.0F, i - 1.0F);
                        float b = TextureHelpers.tex2D(texObj, j + 1.0F, i);

                        float data = (Math.Abs((tl + 2.0F * l + bl - tr - 2.0F * r - br)) +
                                     Math.Abs((tl + 2.0F * t + tr - bl - 2.0F * b - br)));

                        if (data > 255)
                        {
                            data = 255;
                        }

                        TextureHelpers.surf2Dwrite(data, surfObj, j * sizeof(float), i);
                    }
                    else
                    {
                        TextureHelpers.surf2Dwrite(0.0F, surfObj, j * sizeof(float), i);
                    }
                }
            }
        }
    }
}
