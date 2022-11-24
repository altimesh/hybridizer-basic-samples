using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace SimpleCufft
{
    using Complex = float2;

    class Program
    {
        const int SIGNAL_SIZE = 50;
        const int FILTER_KERNEL_SIZE = 11;
        static readonly Random random = new Random(42);

        static void Main(string[] args)
        {
            Complex[] h_signal = new Complex[SIGNAL_SIZE];
            for (int i = 0; i < SIGNAL_SIZE; ++i)
            {
                h_signal[i].x = (float)random.NextDouble();
                h_signal[i].y = 0.0F;
            }

            Complex[] h_filter_kernel = new Complex[FILTER_KERNEL_SIZE];
            for (int i = 0; i < FILTER_KERNEL_SIZE; ++i)
            {
                h_filter_kernel[i].x = (float)random.NextDouble();
                h_filter_kernel[i].y = 0.0F;
            }

            Complex[] h_padded_signal, h_padded_filter_kernel;
            int new_size = PadData(h_signal, out h_padded_signal, SIGNAL_SIZE, h_filter_kernel, out h_padded_filter_kernel, FILTER_KERNEL_SIZE);
            int mem_size = new_size * 2 * sizeof(float);

            // pin host data an copy to device
            GCHandle signalHandle = GCHandle.Alloc(h_padded_signal, GCHandleType.Pinned);
            GCHandle kernelHandle = GCHandle.Alloc(h_padded_filter_kernel, GCHandleType.Pinned);

            IntPtr d_signal;
            cuda.ERROR_CHECK(cuda.Malloc(out d_signal, mem_size));
            IntPtr d_filter_kernel;
            cuda.ERROR_CHECK(cuda.Malloc(out d_filter_kernel, mem_size));

            cuda.ERROR_CHECK(cuda.Memcpy(d_signal, signalHandle.AddrOfPinnedObject(), mem_size, cudaMemcpyKind.cudaMemcpyHostToDevice));
            cuda.ERROR_CHECK(cuda.Memcpy(d_filter_kernel, kernelHandle.AddrOfPinnedObject(), mem_size, cudaMemcpyKind.cudaMemcpyHostToDevice));


            cufftHandle plan;
            cufft_ERROR_CHECK(cufft.Plan1d(out plan, new_size, cufftType.CUFFT_C2C, 1));

            cufftHandle plan_adv;
            size_t workSize;
            long new_size_long = new_size;
            cufft_ERROR_CHECK(cufft.Create(out plan_adv));
            cufft_ERROR_CHECK(cufft.XtMakePlanMany(plan_adv, 1, new long[1] { new_size_long }, null, 1, 1, cudaDataType_t.CUDA_C_32F, null, 1, 1, cudaDataType_t.CUDA_C_32F, 1, out workSize, cudaDataType_t.CUDA_C_32F));

            cufft_ERROR_CHECK(cufft.ExecC2C(plan, d_signal, d_signal, -1));
            cufft_ERROR_CHECK(cufft.ExecC2C(plan_adv, d_filter_kernel, d_filter_kernel, -1));

            dynamic wrapped = HybRunner.Cuda().SetDistrib(32, 256).Wrap(new Program());
            wrapped.ComplexPointwiseMulAndScale(d_signal, d_filter_kernel, new_size, 1.0F / new_size);

            cuda.ERROR_CHECK(cuda.GetLastError());
            
            cufft_ERROR_CHECK(cufft.ExecC2C(plan, d_signal, d_signal, 1));

            cuda.ERROR_CHECK(cuda.Memcpy(signalHandle.AddrOfPinnedObject(), d_signal, mem_size, cudaMemcpyKind.cudaMemcpyDeviceToHost));
            
            Complex[] h_convolved_signal_ref = new Complex[SIGNAL_SIZE];
            Convolve(h_signal, SIGNAL_SIZE, h_filter_kernel, FILTER_KERNEL_SIZE, h_convolved_signal_ref);

            bool bResult = CompareL2(h_convolved_signal_ref, h_padded_signal, 1.0E-5F);

            cufft_ERROR_CHECK(cufft.Destroy(plan));
            cufft_ERROR_CHECK(cufft.Destroy(plan_adv));
            signalHandle.Free();
            kernelHandle.Free();
            cuda.ERROR_CHECK(cuda.Free(d_signal));
            cuda.ERROR_CHECK(cuda.Free(d_filter_kernel));

            if(bResult)
            {
                Console.Error.WriteLine("ERROR");
                Environment.Exit(6);
            }

            Console.Out.WriteLine("OK");
        }

        static void cufft_ERROR_CHECK(cufftResult result)
        {
            if (result != cufftResult.CUFFT_SUCCESS)
            {
                Console.Error.WriteLine(Enum.GetName(typeof(cufftResult), result));
                Environment.Exit(6); // abort;
            }
        }

        static int PadData(Complex[] signal, out Complex[] padded_signal, int signal_size,
            Complex[] filter_kernel, out Complex[] padded_filter_kernel,
            int filter_kernel_size)
        {
            int minRadius = filter_kernel_size / 2;
            int maxRadius = filter_kernel_size - minRadius;
            int new_size = signal_size + maxRadius;

            // Pad signal
            Complex[] new_signal = new Complex[new_size];
            for (int k = 0; k < signal_size; ++k)
            {
                new_signal[k] = signal[k];
            }
            for (int k = 0; k < new_size - signal_size; ++k)
            {
                new_signal[k + signal_size].x = 0.0F;
                new_signal[k + signal_size].y = 0.0F;
            }

            padded_signal = new_signal;


            //// Pad filter
            var new_kernel = new Complex[new_size];
            for (int k = 0; k < maxRadius; ++k)
            {
                new_kernel[k] = filter_kernel[minRadius + k];
            }
            for (int k = 0; k < new_size - filter_kernel_size; ++k)
            {
                new_kernel[maxRadius + k].x = 0.0F;
                new_kernel[maxRadius + k].y = 0.0F;
            }

            padded_filter_kernel = new_kernel;

            return new_size;
        }

        static void Convolve(Complex[] signal, int signal_size,
                Complex[] filter_kernel, int filter_kernel_size,
                Complex[] filtered_signal)
        {
            int minRadius = filter_kernel_size / 2;
            int maxRadius = filter_kernel_size - minRadius;

            // Loop over output element indices
            for (int i = 0; i < signal_size; ++i)
            {
                filtered_signal[i].x = filtered_signal[i].y = 0;

                // Loop over convolution indices
                for (int j = -maxRadius + 1; j <= minRadius; ++j)
                {
                    int k = i + j;

                    if (k >= 0 && k < signal_size)
                    {
                        filtered_signal[i] =
                            filtered_signal[i] + ComplexMul(signal[k], filter_kernel[minRadius - j]);
                    }
                }
            }
        }


        [EntryPoint]
        static void ComplexPointwiseMulAndScale(Complex[] a, Complex[] b, int size, float scale)
        {
            int numThreads = blockDim.x * gridDim.x;
            int threadID = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = threadID; i < size; i += numThreads)
            {
                a[i] = Scale(scale, ComplexMul(a[i], b[i]));
            }
        }

        [Kernel] 
        static Complex Scale(float s, Complex a)
        {
            Complex result = new Complex();
            result.x = a.x * s;
            result.y = a.y * s;
            return result;
        }

        [Kernel]
        static Complex ComplexMul(Complex a, Complex b)
        {
            Complex result = new Complex();
            result.x = a.x * b.x - a.y * b.y;
            result.y = a.x * b.y + a.y * b.x;
            return result;
        }


        static bool CompareL2(Complex[] reference, Complex[] data, float epsilon)
        {
            float error = 0;
            float tmp = 0;

            for (int i = 0; i < reference.Length; ++i)
            {
                float diffX = reference[i].x - data[i].x;
                error += diffX * diffX;
                float diffY = reference[i].y - data[i].y;
                error += diffY * diffY;
                tmp += reference[i].x * reference[i].x + reference[i].y * reference[i].y;
            }

            float normRef = HybMath.Sqrt(tmp);

            if (HybMath.Abs(tmp) < 1e-7)
            {
                return false;
            }

            float normError = HybMath.Sqrt(error);
            error = normError / normRef;
            bool result = error < epsilon;
            return result;
        }
    }
}
