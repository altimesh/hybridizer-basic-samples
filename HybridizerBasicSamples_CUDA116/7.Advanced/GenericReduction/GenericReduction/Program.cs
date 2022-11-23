﻿using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GenericReduction
{
	[HybridTemplateConcept]
	interface IReductor
	{
		[Kernel]
		float func(float x, float y);
		[Kernel]
		float neutral { get; }
	}
	
	struct AddReductor: IReductor
	{
		[Kernel]
		public float neutral { get { return 0.0F; } }

		[Kernel]
		public float func(float x, float y)
		{
			return x + y;
		}
	}

	struct MaxReductor : IReductor
	{
		[Kernel]
		public float neutral { get { return float.MinValue; } }

		[Kernel]
		public float func(float x, float y)
		{
			return Math.Max(x, y);
		}
	}

	[HybridRegisterTemplate(Specialize = typeof(GridReductor<MaxReductor>))]
	[HybridRegisterTemplate(Specialize = typeof(GridReductor<AddReductor>))]
	class GridReductor<TReductor> where TReductor : struct, IReductor
	{
		[Kernel]
		TReductor reductor { get { return default(TReductor); } }

		[Kernel]
		public void Reduce(float[] result, float[] input, int N)
		{
			var cache = new SharedMemoryAllocator<float>().allocate(blockDim.x);
			int tid = threadIdx.x + blockDim.x * blockIdx.x;
			int cacheIndex = threadIdx.x;

			float tmp = reductor.neutral;
			while (tid < N)
			{
				tmp = reductor.func(tmp, input[tid]);
				tid += blockDim.x * gridDim.x;
			}

			cache[cacheIndex] = tmp;

			CUDAIntrinsics.__syncthreads();

			int i = blockDim.x / 2;
			while (i != 0)
			{
				if (cacheIndex < i)
				{
					cache[cacheIndex] = reductor.func(cache[cacheIndex], cache[cacheIndex + i]);
				}

				CUDAIntrinsics.__syncthreads();
				i >>= 1;
			}

			if (cacheIndex == 0)
			{
				AtomicExpr.apply(ref result[0], cache[0], reductor.func);
			}
		}
	}

	// Unfortunately this is necessay since we didn't implemented generic entrypoints yet. 
	class EntryPoints
	{
		[EntryPoint]
		public static void ReduceAdd(GridReductor<AddReductor> reductor, float[] result, float[] input, int N)
		{
			reductor.Reduce(result, input, N);
		}
		[EntryPoint]
		public static void ReduceMax(GridReductor<MaxReductor> reductor, float[] result, float[] input, int N)
		{
			reductor.Reduce(result, input, N);
		}
	}

	class Program
	{
		static void Main(string[] args)
		{
			const int N = 1024 * 1024 * 32;
			float[] a = new float[N];

			// initialization
			Random random = new Random(42);
			Parallel.For(0, N, i => a[i] = (float)random.NextDouble());

			// hybridizer configuration
			cudaDeviceProp prop;
			cuda.GetDeviceProperties(out prop, 0);
			int gridDimX = 16 * prop.multiProcessorCount;
            int blockDimX = 256;
            cuda.DeviceSetCacheConfig(cudaFuncCache.cudaFuncCachePreferShared);
			HybRunner runner = HybRunner.Cuda().SetDistrib(gridDimX, 1, blockDimX, 1, 1, blockDimX * sizeof(float));
			float[] buffMax = new float[1];
			float[] buffAdd = new float[1];
			var maxReductor = new GridReductor<MaxReductor>();
			var addReductor = new GridReductor<AddReductor>();
			dynamic wrapped = runner.Wrap(new EntryPoints());

			// device reduction
			wrapped.ReduceMax(maxReductor, buffMax, a, N);
			wrapped.ReduceAdd(addReductor, buffAdd, a, N);
			cuda.ERROR_CHECK(cuda.DeviceSynchronize());

			// check results
			float expectedMax = a.AsParallel().Aggregate((x, y) => Math.Max(x, y));
			float expectedAdd = a.AsParallel().Aggregate((x, y) => x + y);
			bool hasError = false;
			if (buffMax[0] != expectedMax)
			{
				Console.Error.WriteLine($"MAX Error : {buffMax[0]} != {expectedMax}");
				hasError = true;
			}

			// addition is not associative, so results cannot be exactly the same
			// https://en.wikipedia.org/wiki/Associative_property#Nonassociativity_of_floating_point_calculation
			if (Math.Abs(buffAdd[0] - expectedAdd) / expectedAdd > 1.0E-5F)
			{
				Console.Error.WriteLine($"ADD Error : {buffAdd[0]} != {expectedAdd}");
				hasError = true;
			}

			if (hasError)
				Environment.Exit(1);

			Console.Out.WriteLine("OK");
		}
	}
}
