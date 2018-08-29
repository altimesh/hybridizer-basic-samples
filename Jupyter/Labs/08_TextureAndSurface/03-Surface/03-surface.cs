using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Hybridizer.Runtime.CUDAImports;

namespace TextureAndSurface
{
    class Program
    {
        static void Main(string[] args)
        {
            HybRunner runner = HybRunner.Cuda().SetDistrib(32, 32, 16, 16, 1, 0);
            
            GrayBitmap image = GrayBitmap.Load("../../images/lena512.bmp");
            uint height = image.Height, width = image.Width;
            ushort[] inputPixels = image.PixelsUShort;
            
            float[] imageFloat = new float[width * height];
            float[] imageCompute = new float[width * height];
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
            
            //create and bind surface
            
            dynamic wrapper = runner.Wrap(new Program());

            wrapper.Sobel(texObj, imageCompute, (int)width, (int)height);

            ushort[] outputPixel = new ushort[width * height];
            for (int i = 0; i < width * height; ++i)
            {
                outputPixel[i] = (ushort)imageCompute[i];
            }
            
            GrayBitmap imageSobel = new GrayBitmap(width, height);
            imageSobel.PixelsUShort = outputPixel;
            imageSobel.Save("../../output-03-surface/sobel.bmp");
        }

        [EntryPoint]
        public static void Sobel(cudaTextureObject_t texObj, float[] output, int width, int height) //make output to be a surface
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

                        output[pixelId] = data;
                    }
                    else
                    {
                        output[pixelId] = 0;
                    }
                }
            }
        }
    }
}
