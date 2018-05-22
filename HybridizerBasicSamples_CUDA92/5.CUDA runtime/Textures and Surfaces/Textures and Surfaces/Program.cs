using Hybridizer.Runtime.CUDAImports;
using System;
using System.Drawing;
using System.Runtime.InteropServices;

namespace Textures_and_Surfaces
{
    class Program
    {
        unsafe static void Main(string[] args)
        {
            HybRunner runner = HybRunner.Cuda(@"Textures and Surfaces_CUDA.dll").SetDistrib(32, 32, 16, 16, 1, 0);
            Bitmap baseImage = (Bitmap)Image.FromFile("lena512.bmp");
            int height = baseImage.Height, width = baseImage.Width;

            byte[] inputPixels = new byte[width * height];
            ReadImage(inputPixels, baseImage, width, height);
            float[] imagefloat = new float[width * height];
            for (int i = 0; i < width * height; ++i)
            {
                imagefloat[i] = (float)inputPixels[i];
            }

            IntPtr src = runner.Marshaller.MarshalManagedToNative(imagefloat);

            //bind texture
            cudaChannelFormatDesc channelDescTex = TextureHelpers.cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindFloat);
            cudaArray_t cuArrayTex = TextureHelpers.CreateCudaArray(channelDescTex, src, width, height);
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

            //create cudaResourceDesc
            cudaResourceDesc resDescSurf = TextureHelpers.CreateCudaResourceDesc(cuArraySurf);

            //create surface object
            cudaSurfaceObject_t surfObj;
            cuda.CreateSurfaceObject(out surfObj, ref resDescSurf);

            dynamic wrapper = runner.Wrap(new Program());

            // call kernek
            wrapper.Sobel(texObj, surfObj, width, height);


            float[] imageSobel = new float[width * height];
            for (int i = 0; i < width * height; ++i)
            {
                imageSobel[i] = 128.0F;
            }
            GCHandle handle = GCHandle.Alloc(imageSobel, GCHandleType.Pinned);
            IntPtr dest = handle.AddrOfPinnedObject();

            cuda.MemcpyFromArray(dest, cuArraySurf, 0, 0, width * height * sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost);

            byte[] imageSobelByte = new byte[width * height];
            for (int i = 0; i < width * height; ++i)
            {
                imageSobelByte[i] = (byte)imageSobel[i];
            }

            SaveImage("lenaSobel512.bmp", imageSobelByte, width, height);
            cuda.DestroySurfaceObject(surfObj);
            cuda.DestroyTextureObject(texObj);
        }

        [IntrinsicFunction("fabsf")]
        public static float fabsf(float f)
        {
            return Math.Abs(f);
        }

        public static void ReadImage(byte[] inputPixel, Bitmap image, int width, int height)
        {
            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    inputPixel[i * height + j] = image.GetPixel(i, j).B;
                }
            }
        }

        [EntryPoint]
        public static void Sobel(cudaTextureObject_t texObj, cudaSurfaceObject_t surfObj, int width, int height)
        {
            float ii, jj;
            for (int i = threadIdx.y + blockIdx.y * blockDim.y; i < height; i += blockDim.y * gridDim.y)
            {
                ii = (float)i;
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < width; j += blockDim.x * gridDim.x)
                {
                    if (i != 0 && j != 0 && i != height - 1 && j != width - 1)
                    {
                        jj = (float)j;
                        float tl = TextureHelpers.tex2D(texObj, jj - 1.0F, ii - 1.0F);
                        float t = TextureHelpers.tex2D(texObj, jj - 1.0F, ii);
                        float tr = TextureHelpers.tex2D(texObj, jj - 1.0F, ii + 1.0F);
                        float l = TextureHelpers.tex2D(texObj, jj, ii - 1.0F);
                        float r = TextureHelpers.tex2D(texObj, jj, ii + 1.0F);
                        float br = TextureHelpers.tex2D(texObj, jj + 1.0F, ii + 1.0F);
                        float bl = TextureHelpers.tex2D(texObj, jj + 1.0F, ii - 1.0F);
                        float b = TextureHelpers.tex2D(texObj, jj + 1.0F, ii);

                        float data = fabsf((tl + 2.0F * l + bl - tr - 2.0F * r - br)) +
                                     fabsf((tl + 2.0F * t + tr - bl - 2.0F * b - br));

                        if (data > 255.0F)
                        {
                            data = 255.0F;
                        }

                        TextureHelpers.surf2Dwrite(data, surfObj, j * sizeof(int), i);
                    }
                    else
                    {
                        TextureHelpers.surf2Dwrite(0.0F, surfObj, j * sizeof(int), i);
                    }
                }
            }
        }

        public static void SaveImage(string nameImage, byte[] outputPixel, int width, int height)
        {
            Bitmap resImage = new Bitmap(width, height);
            int col = 0;
            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    col = outputPixel[i * height + j];
                    resImage.SetPixel(i, j, Color.FromArgb(col, col, col));
                }
            }

            //store the result image
            resImage.Save(nameImage, System.Drawing.Imaging.ImageFormat.Bmp);
        }
    }
}
