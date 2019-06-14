using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

using Hybridizer.Imaging;
using Hybridizer.Runtime.CUDAImports;

namespace MedianFilter
{
    public class Filter
    {
        const int window = 3 ;
        const int windowCount = window + window + 1 ;
            
        [IntrinsicInclude("intrinsics.cuh")]
        [IntrinsicType("medianfilter<unsigned short, 3>")]
        struct medianfilter_ushort_3
        {
            public ushort apply() { throw new NotImplementedException(); }
            public void rollbuffer() { throw new NotImplementedException(); }
            public ushort get_Item(int i) { throw new NotImplementedException(); }
            public void set_Item(int i, ushort val) { throw new NotImplementedException(); }
        }
        
        [EntryPoint]
        public static void ParForGPU(ushort[] output, ushort[] input, int width, int height, int chunk)
        {
            int i,j ;
            medianfilter_ushort_3 filter;
            for (int bj = blockIdx.y * chunk; bj < height; bj += gridDim.y * chunk)
            {
                for (int bi = threadIdx.x + blockDim.x * blockIdx.x; bi < width; bi += blockDim.x * gridDim.x)
                {
                    // preload window
                    for (int lj = -window; lj < window; ++lj)
                    {
                        j = bj + lj;
                        if (j < 0) j = 0;
                        if (j >= height) j = height - 1;
                        
                        for (int li = -window; li <= window; ++li)
                        {
                            i = bi + li;
                            if (i < 0) i = 0;
                            if (i >= width) i = width - 1;

                            filter.set_Item((lj + window) * (window * 2 + 1) + (li + window), input[j * width + i]);
                        }
                    }

                    // process
                    for (int lj = 0; lj < chunk; ++lj)
                    {
                        // roll the buffer
                        if (lj != 0) filter.rollbuffer();

                        // load next line
                        j = bj + lj + window;
                        if (j < 0) j = 0;
                        if (j >= height) j = height - 1;
                        
                        for (int li = -window; li <= window; ++li)
                        {
                            i = bi + li;
                            if (i < 0) i = 0;
                            if (i >= width) i = width - 1;

                            filter.set_Item((2 * window) * (window * 2 + 1) + (li + window), input[j * width + i]);
                        }

                        // process buffer
                        ushort value = filter.apply();

                        j = bj + lj;
                        if (j < 0) j = 0;
                        if (j >= height) j = height - 1;

                        // store output
                        output[j * width + bi] = value;
                    }
                }
            }
        }
    }
    
    class Program
    {
        
        static void Main(string[] args)
        {
            int currentDevice ;
            cuda.GetDevice(out currentDevice) ;
            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, currentDevice) ;
            
            GrayBitmap image = GrayBitmap.Load("../../images/lena_highres_greyscale_noise.bmp");
            GrayBitmap denoised = new GrayBitmap(image.Width, image.Height) ;
            ushort[] input = image.PixelsUShort ;
            ushort[] output = new ushort[image.Width * image.Height];
            
            Stopwatch watch = new Stopwatch();
            watch.Start();
            
            int chunk ;
            if ((prop.major >= 6) && (prop.minor == 0))
                chunk = ((int)image.Height + (prop.multiProcessorCount/2) -1) / (prop.multiProcessorCount/2) ;
            else
                chunk = ((int)image.Height + (prop.multiProcessorCount) -1) / (prop.multiProcessorCount) ;
            
            Console.Out.WriteLine("Chunk size = {0}", chunk) ;

            dim3 grid = new dim3(16, ((int)image.Height + chunk-1)/chunk,1) ;
            dim3 block = new dim3(128, 1,1) ;
            
            // create an instance of runner
            HybRunner runner = HybRunner.Cuda();
            // wrap a new instance of Program
            dynamic wrapper = runner.Wrap(new Filter());
            // run the method on GPU
            wrapper.SetDistrib(grid,block).ParForGPU (output, input, (int)image.Width, (int)image.Height, chunk) ;
                        
            watch.Stop();
            string time = String.Format("{0:0.00}", watch.ElapsedMilliseconds * 1.0E-3);
            
            string kernelTime = String.Format("{0:0.00}", runner.LastKernelDuration.ElapsedMilliseconds * 1.0E-3);

            Console.WriteLine ($"SweepSort GPU time : {time}");
            Console.WriteLine ($"SweepSort GPU -- kernel time : {kernelTime}");
            denoised.PixelsUShort = output ;
            denoised.Save ("../../output-07-cache-aware-gpu/denoised.bmp");
        }
    }
}
        