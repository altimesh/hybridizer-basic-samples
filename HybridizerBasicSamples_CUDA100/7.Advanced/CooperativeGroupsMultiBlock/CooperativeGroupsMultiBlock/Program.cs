﻿using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using cg = Hybridizer.Runtime.CUDAImports.cooperative_groups;

namespace CooperativeGroupsMultiBlock
{
    class Program
    {
        [Kernel]
        static void reduceBlock(double[] sdata, thread_block cta)
        {
            uint tid = cta.thread_rank();
            thread_block_tile_32 tile32 = cg.tile_partition_32(cta);

            double beta = sdata[tid];
            double temp;

            for (uint i = tile32.size() / 2; i > 0; i >>= 1)
            {
                if (tile32.thread_rank() < i)
                {
                    temp = sdata[tid + i];
                    beta += temp;
                    sdata[tid] = beta;
                }
                tile32.sync();
            }
            cta.sync();

            if (cta.thread_rank() == 0)
            {
                beta = 0;
                for (uint i = 0; i < blockDim.x; i += tile32.size())
                {
                    beta += sdata[i];
                }
                sdata[0] = beta;
            }
            cta.sync();
        }

        [EntryPoint]
        public static void reduceSinglePassMultiBlockCG(float[] g_idata, float[] g_odata, uint n)
        {
            // Handle to thread block group
            thread_block block = cg.this_thread_block();
            grid_group grid = cg.this_grid();

            double[] sdata = new SharedMemoryAllocator<double>().allocate(blockDim.x);

            // Stride over grid and add the values to a shared memory buffer
            sdata[block.thread_rank()] = 0;

            for (uint i = grid.thread_rank(); i < n; i += grid.size())
            {
                sdata[block.thread_rank()] += g_idata[i];
            }

            block.sync();

            // Reduce each block (called once per block)
            reduceBlock(sdata, block);
            // Write out the result to global memory
            if (block.thread_rank() == 0)
            {
                g_odata[blockIdx.x] = (float)sdata[0];
            }

            grid.sync();

            if (grid.thread_rank() == 0)
            {
                for (uint blockId = 1; blockId < gridDim.x; blockId++)
                {
                    g_odata[0] += g_odata[blockId];
                }
            }
        }

        static void Main(string[] args)
        {
            // TODO: enable cudaOccupancyCalculator
            int deviceCount;
            int numBlocks = 32; // 4 * prop.multiProcessorCount;
            int numThreads = 128;
            cuda.GetDeviceCount(out deviceCount);
            bool found = false;
            for (int i = 0; i < deviceCount; ++i)
            {
                cudaDeviceProp prop;
                cuda.GetDeviceProperties(out prop, i);
                if (prop.cooperativeLaunch != 0)
                {
                    cuda.SetDevice(i);
                    numBlocks = 1 * prop.multiProcessorCount;
                    numThreads = 128;
                    Console.Out.WriteLine($"running on device {i}");
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                Console.Error.WriteLine("No GPU Found supporting Cooperative Launch");
                Environment.Exit(6);
            }

            const int N = 1024 * 1024;

            float[] a = new float[N];
            float[] b = new float[numBlocks];

            for (int i = 0; i < N; ++i)
            {
                a[i] = 1.0F;
            }
            var runner = HybRunner.Cuda().SetGridSync(true);
            dynamic wrapped = runner.SetDistrib(numBlocks, 1, numThreads, 1, 1, numThreads * sizeof(double)).Wrap(new Program());
            runner.saveAssembly();
            cuda.ERROR_CHECK((cudaError_t)(int)wrapped.reduceSinglePassMultiBlockCG(a, b, N));
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
            Console.Out.WriteLine(String.Join(", ", b.Take(32)));
        }
    }
}
