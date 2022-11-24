using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NPP_ImageSegmentation
{
    class Program
    {
        const string inputFileName = @"large_objects.png";
        const string segmentsFileName = @"objects_segments.png";

        [DllImport("NPP_ImageSegmentation_CUDA.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int NPP_ImageSegmentationx46Programx46ColorizeLabels_ExternCWrapperStream_CUDA(
            int gridDimX, int gridDimY, int blockDimX, int blockDimY, int blockDimZ, int shared, cudaStream_t stream,
            IntPtr segments, IntPtr colors, IntPtr colormap, int maxLabel, int count, int width, int pitch);

        [EntryPoint]
        public static void ColorizeLabels(ushort[] segments, uchar4[] colors, uchar4[] colormap, int maxLabel, int count, int width, int pitch)
        {
            for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < count; tid += blockDim.x * gridDim.x)
            {
                int colorsI = tid % width;
                if (colorsI < width)
                {
                    int segmentsI = tid % pitch;
                    int segmentsJ = tid / pitch;
                    int colorsJ = tid / width;
                    ushort label = segments[tid];
                    if (label != 0)
                    {
                        colors[segmentsJ * width + segmentsI] = colormap[label];
                    }
                }
            }
        }


        static void Main(string[] args)
        {
            // init CUDA
            IntPtr d;
            cuda.Malloc(out d, sizeof(int));
            cuda.Free(d);

            HybRunner runner = HybRunner.Cuda();
            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);
            dynamic wrapped = runner.Wrap(new Program());
            runner.saveAssembly();
            cudaStream_t stream;
            cuda.StreamCreate(out stream);

            NppStreamContext context = new NppStreamContext
            {
                hStream = stream,
                nCudaDevAttrComputeCapabilityMajor = prop.major,
                nCudaDevAttrComputeCapabilityMinor = prop.minor,
                nCudaDeviceId = 0,
                nMaxThreadsPerBlock = prop.maxThreadsPerBlock,
                nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor,
                nMultiProcessorCount = prop.multiProcessorCount,
                nSharedMemPerBlock = 0
            };

            Random rand = new Random();

            using (NPPImage input = NPPImage.Load(inputFileName, stream))
            {
                uchar4[] output = new uchar4[input.width * input.height];
                IntPtr d_output;
                cuda.Malloc(out d_output, input.width * input.height * 4 * sizeof(byte));

                // working area 
                IntPtr oDeviceDst32u;
                size_t oDeviceDst32uPitch;
                cuda.ERROR_CHECK(cuda.MallocPitch(out oDeviceDst32u, out oDeviceDst32uPitch, input.width * sizeof(int), input.height));
                IntPtr segments;
                size_t segmentsPitch;
                cuda.ERROR_CHECK(cuda.MallocPitch(out segments, out segmentsPitch, input.width * sizeof(ushort), input.height));

                NppiSize oSizeROI = new NppiSize { width = input.width, height = input.height };
                int nBufferSize = 0;
                IntPtr pScratchBufferNPP1, pScratchBufferNPP2;

                // compute maximum label
                NPPI.ERROR_CHECK(NPPI.LabelMarkersGetBufferSize_16u_C1R(oSizeROI, out nBufferSize));
                cuda.ERROR_CHECK(cuda.Malloc(out pScratchBufferNPP1, nBufferSize));
                int maxLabel;
                NPPI.ERROR_CHECK(NPPI.LabelMarkers_16u_C1IR_Ctx(input.deviceData, input.pitch, oSizeROI, 165, NppiNorm.nppiNormInf, out maxLabel, pScratchBufferNPP1, context));


                // compress labels
                NPPI.ERROR_CHECK(NPPI.CompressMarkerLabelsGetBufferSize_16u_C1R(maxLabel, out nBufferSize));
                cuda.ERROR_CHECK(cuda.Malloc(out pScratchBufferNPP2, nBufferSize));
                NPPI.ERROR_CHECK(NPPI.CompressMarkerLabels_16u_C1IR_Ctx(input.deviceData, input.pitch, oSizeROI, maxLabel, out maxLabel, pScratchBufferNPP2, context));

                uchar4[] colormap = new uchar4[maxLabel + 1];
                for (int i = 0; i <= maxLabel; ++i)
                {
                    colormap[i] = new uchar4 { x = (byte)(rand.Next() % 256), y = (byte)(rand.Next() % 256), z = (byte)(rand.Next() % 256), w = 0 };
                }

                IntPtr d_colormap;
                cuda.Malloc(out d_colormap, (maxLabel + 1) * 4 * sizeof(byte));
                var handle = GCHandle.Alloc(colormap, GCHandleType.Pinned);
                cuda.Memcpy(d_colormap, handle.AddrOfPinnedObject(), (maxLabel + 1) * 4 * sizeof(byte), cudaMemcpyKind.cudaMemcpyHostToDevice);
                handle.Free();

                NPP_ImageSegmentationx46Programx46ColorizeLabels_ExternCWrapperStream_CUDA(
                    8 * prop.multiProcessorCount, 1, 256, 1, 1, 0, stream, // cuda configuration
                    input.deviceData, d_output, d_colormap, maxLabel + 1, input.pitch * input.height / sizeof(ushort), input.width, input.pitch / sizeof(ushort));

                handle = GCHandle.Alloc(output, GCHandleType.Pinned);
                cuda.Memcpy(handle.AddrOfPinnedObject(), d_output, input.width * input.height * sizeof(byte) * 4, cudaMemcpyKind.cudaMemcpyDeviceToHost);
                handle.Free();
                NPPImage.Save(segmentsFileName, output, input.width, input.height);
                Process.Start(segmentsFileName);

                cuda.ERROR_CHECK(cuda.Free(oDeviceDst32u));
                cuda.ERROR_CHECK(cuda.Free(segments));
                cuda.ERROR_CHECK(cuda.Free(pScratchBufferNPP1));
                cuda.ERROR_CHECK(cuda.Free(pScratchBufferNPP2));
            }
        }
    }
}
