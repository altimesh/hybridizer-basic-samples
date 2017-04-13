using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace BlackScholesFloat4
{
    class Program
    {
        const int OPT_N = 4000000;
        const int NUM_ITERATIONS = 512;

        const int OPT_SZ = OPT_N * sizeof(float);
        const float RISKFREE = 0.02f;
        const float VOLATILITY = 0.30f;

        static void Main(string[] args)
        {
            float4[] callResult_net = new float4[OPT_N/4];
            float4[] putResult_net = new float4[OPT_N/4];
            float4[] stockPrice_net = new float4[OPT_N/4];
            float4[] optionStrike_net = new float4[OPT_N/4];
            float4[] optionYears_net = new float4[OPT_N/4];

            float4[] callResult_cuda = new float4[OPT_N/4];
            float4[] putResult_cuda = new float4[OPT_N/4];

            Stopwatch watch = new Stopwatch();
            Random rand = new Random(Guid.NewGuid().GetHashCode());
            for (int i = 0; i < OPT_N/4; ++i)
            {
                callResult_net[i] = new float4(0.0f, 0.0f, 0.0f, 0.0f);
                putResult_net[i] = new float4(-1.0f, -1.0f, -1.0f, -1.0f) ;
                stockPrice_net[i] = new float4((float)rand.NextDouble() * 25.0f + 5.0f,
                                                 (float)rand.NextDouble() * 25.0f + 5.0f,
                                                 (float)rand.NextDouble() * 25.0f + 5.0f,
                                                 (float)rand.NextDouble() * 25.0f + 5.0f);
                optionStrike_net[i] = new float4((float)rand.NextDouble() * 99.0f + 1.0f,
                                                   (float)rand.NextDouble() * 99.0f + 1.0f,
                                                   (float)rand.NextDouble() * 99.0f + 1.0f,
                                                   (float)rand.NextDouble() * 99.0f + 1.0f);
                optionYears_net[i] = new float4((float)rand.NextDouble() * 9.75f + 0.25f,
                                                (float)rand.NextDouble() * 9.75f + 0.25f,
                                                (float)rand.NextDouble() * 9.75f + 0.25f,
                                                (float)rand.NextDouble() * 9.75f + 0.25f);
            }

            HybRunner runner = HybRunner.Cuda("BlackScholesFloat4_CUDA.dll").SetDistrib(20, 256);
            dynamic wrapper = runner.Wrap(new Program());

            watch.Start();
            for (int i = 0; i < NUM_ITERATIONS; ++i)
            {
                wrapper.BlackScholes(callResult_cuda,
                             putResult_cuda,
                             stockPrice_net,
                             optionStrike_net,
                             optionYears_net,
                             0, OPT_N/4);
            }
            watch.Stop();
            Console.WriteLine("nb ms cuda     : {0}", watch.ElapsedMilliseconds / NUM_ITERATIONS);
            Console.WriteLine("without memcpy : {0}", runner.LastKernelDuration.ElapsedMilliseconds);

            watch.Restart();
            for (int i = 0; i < NUM_ITERATIONS; ++i)
            {
                Parallel.For(0, OPT_N/4, (opt) =>
                {
                    BlackScholes(callResult_net,
                                 putResult_net,
                                 stockPrice_net,
                                 optionStrike_net,
                                 optionYears_net,
                                 opt,
                                 opt + 1);
                });
            }
            watch.Stop();
            Console.WriteLine("nb ms c#       : {0}", watch.ElapsedMilliseconds / NUM_ITERATIONS);

            WriteCalculationError(callResult_net, callResult_cuda, putResult_net, putResult_cuda);

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("fabsf")]
        public static float fabsf(float f)
        {
            return Math.Abs(f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("expf")]
        public static float Expf(float f)
        {
            return (float)Math.Exp((double)f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("sqrtf")]
        public static float Sqrtf(float f)
        {
            return (float)Math.Sqrt((double)f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining), IntrinsicFunction("logf")]
        public static float Logf(float f)
        {
            return (float)Math.Log((double)f);
        }

        [EntryPoint]
        public static void BlackScholes(
            float4[] callResult,
            float4[] putResult,
            float4[] stockPrice,
            float4[] optionStrike,
            float4[] optionYears,
            int lineFrom,
            int lineTo
            )
        {
            for (int i = lineFrom + blockDim.x * blockIdx.x + threadIdx.x; i < lineTo; i += blockDim.x * gridDim.x)
            {
                #region blackscholesx
                float sqrtTX, expRTX;
                float f1X, f2X, CNDF1X, CNDF2X;

                sqrtTX = Sqrtf(optionYears[i].x);
                f1X = (Logf(stockPrice[i].x / optionStrike[i].x) + (RISKFREE + 0.5f * VOLATILITY * VOLATILITY) * optionYears[i].x) /
                         (VOLATILITY * sqrtTX);
                f2X = f1X - VOLATILITY * sqrtTX;

                CNDF1X = CND(f1X);
                CNDF2X = CND(f2X);

                expRTX = Expf(-RISKFREE * optionYears[i].x);
                callResult[i].x = stockPrice[i].x * CNDF1X - optionStrike[i].x * expRTX * CNDF2X;
                putResult[i].x = optionStrike[i].x * expRTX * (1.0f - CNDF2X) - stockPrice[i].x * (1.0f - CNDF1X);
                #endregion

                #region blackscholesy
                float sqrtTY, expRTY;
                float f1Y, f2Y, CNDF1Y, CNDF2Y;

                sqrtTY = Sqrtf(optionYears[i].y);
                f1Y = (Logf(stockPrice[i].y / optionStrike[i].y) + (RISKFREE + 0.5f * VOLATILITY * VOLATILITY) * optionYears[i].y) /
                         (VOLATILITY * sqrtTY);
                f2Y = f1Y - VOLATILITY * sqrtTY;

                CNDF1Y = CND(f1Y);
                CNDF2Y = CND(f2Y);

                expRTY = Expf(-RISKFREE * optionYears[i].y);
                callResult[i].y = stockPrice[i].y * CNDF1Y - optionStrike[i].y * expRTY * CNDF2Y;
                putResult[i].y = optionStrike[i].y * expRTY * (1.0f - CNDF2Y) - stockPrice[i].y * (1.0f - CNDF1Y);
                #endregion

                #region blackscholesz
                float sqrtTZ, expRTZ;
                float f1Z, f2Z, CNDF1Z, CNDF2Z;

                sqrtTZ = Sqrtf(optionYears[i].z);
                f1Z = (Logf(stockPrice[i].z / optionStrike[i].z) + (RISKFREE + 0.5f * VOLATILITY * VOLATILITY) * optionYears[i].z) /
                         (VOLATILITY * sqrtTZ);
                f2Z = f1Z - VOLATILITY * sqrtTZ;

                CNDF1Z = CND(f1Z);
                CNDF2Z = CND(f2Z);

                expRTZ = Expf(-RISKFREE * optionYears[i].z);
                callResult[i].z = stockPrice[i].z * CNDF1Z - optionStrike[i].z * expRTZ * CNDF2Z;
                putResult[i].z = optionStrike[i].z * expRTZ * (1.0f - CNDF2Z) - stockPrice[i].z * (1.0f - CNDF1Z);
                #endregion

                #region blackscholesw
                float sqrtTW, expRTW;
                float f1W, f2W, CNDF1W, CNDF2W;

                sqrtTW = Sqrtf(optionYears[i].w);
                f1W = (Logf(stockPrice[i].w / optionStrike[i].w) + (RISKFREE + 0.5f * VOLATILITY * VOLATILITY) * optionYears[i].w) /
                         (VOLATILITY * sqrtTW);
                f2W = f1W - VOLATILITY * sqrtTW;

                CNDF1W = CND(f1W);
                CNDF2W = CND(f2W);

                expRTW = Expf(-RISKFREE * optionYears[i].w);
                callResult[i].w = stockPrice[i].w * CNDF1W - optionStrike[i].w * expRTW * CNDF2W;
                putResult[i].w = optionStrike[i].w * expRTW * (1.0f - CNDF2W) - stockPrice[i].w * (1.0f - CNDF1W);
                #endregion

            }
        }

        [Kernel]
        static float CND(float f)
        {
            const float A1 = 0.31938153f;
            const float A2 = -0.356563782f;
            const float A3 = 1.781477937f;
            const float A4 = -1.821255978f;
            const float A5 = 1.330274429f;
            const float RSQRT2PI = 0.39894228040143267793994605993438f;

            float K = 1.0f / (1.0f + 0.2316419f * fabsf(f));

            float cnd = RSQRT2PI * Expf(-0.5f * f * f) *
                        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

            if (f > 0)
                cnd = 1.0f - cnd;

            return cnd;
        }

        public static void WriteCalculationError(float4[] callResult1, float4[] callResult2, float4[] putResult1, float4[] putResult2)
        {
            float maxCallError = 0.0F;
            float maxPutError = 0.0F;
            float callL2Error = 0.0F;
            float putL2Error = 0.0F;
            float callL1Error = 0.0F;
            float putL1Error = 0.0F;

            for (int i = 0; i < OPT_N / 4; ++i)
            {
                float4 tmpCallResult = callResult1[i] - callResult2[i];
                float4 tmpPutResult = putResult1[i] - putResult2[i];

                float callErrorX = fabsf(tmpCallResult.x) ;
                float callErrorY = fabsf(tmpCallResult.y) ;
                float callErrorZ = fabsf(tmpCallResult.z) ;
                float callErrorW = fabsf(tmpCallResult.w) ;

                float putErrorX = fabsf(tmpPutResult.x) ;
                float putErrorY = fabsf(tmpPutResult.y) ;
                float putErrorZ = fabsf(tmpPutResult.z) ;
                float putErrorW = fabsf(tmpPutResult.w) ;

                callL2Error += callErrorX * callErrorX + callErrorY * callErrorY + callErrorZ * callErrorZ + callErrorW * callErrorW;

                putL2Error += putErrorX * putErrorX + putErrorY * putErrorY + putErrorZ * putErrorZ + putErrorW * putErrorW;

                callL1Error += callErrorX + callErrorY + callErrorZ + callErrorW;

                putL1Error += putErrorX + putErrorY + putErrorZ + putErrorW;

                maxCallError = maxCallError > callErrorX ? maxCallError : callErrorX;
                maxCallError = maxCallError > callErrorY ? maxCallError : callErrorY;
                maxCallError = maxCallError > callErrorZ ? maxCallError : callErrorZ;
                maxCallError = maxCallError > callErrorW ? maxCallError : callErrorW;

                maxPutError = maxPutError > putErrorX ? maxPutError : putErrorX;
                maxPutError = maxPutError > putErrorY ? maxPutError : putErrorY;
                maxPutError = maxPutError > putErrorZ ? maxPutError : putErrorZ;
                maxPutError = maxPutError > putErrorW ? maxPutError : putErrorW;
            }

            callL1Error /= (float)OPT_N;
            putL1Error /= (float)OPT_N;
            callL2Error = Sqrtf(callL2Error) / (float)OPT_N;
            putL2Error = Sqrtf(putL2Error) / (float)OPT_N;

            Console.WriteLine("CALL ERRORS : Linf : {0:G17}, L2 : {1:G17}, L1: {2:G17}", maxCallError, callL2Error, callL1Error);
            Console.WriteLine("PUT ERRORS  : Linf : {0:G17}, L2 : {1:G17}, L1: {2:G17}", maxPutError, putL2Error, putL1Error);
        }
    }
}
