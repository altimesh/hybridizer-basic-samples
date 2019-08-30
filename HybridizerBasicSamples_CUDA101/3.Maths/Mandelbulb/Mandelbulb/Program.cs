using Hybridizer.Runtime.CUDAImports;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Input;
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;

namespace Mandelbulb
{
    public class Program : GameWindow
    {
        int textureID, shaderProgram, quadVAO;
        cudaSurfaceObject_t surface;

        public static float lightAngle = 140.0F;
        public static float viewAngle = 150.0F;
        public static float eyeDistanceFromNearField = 2.2F;
        public float depth_of_field;
        public float3 viewDirection;
        public float3 nearFieldLocation;
        public float3 eyeLocation;
        public float3 lightDirection;
        public int iterations = 300;
        HybRunner runner;
        dynamic wrapped;

        private void Init()
        {
            depth_of_field = 2.5F;
            float rad = MathFunctions.toRad(lightAngle);
            var lightX = MathFunctions.cosf(rad) * depth_of_field / 2;
            var lightZ = MathFunctions.sinf(rad) * depth_of_field / 2;

            float3 lightLocation = float3.make_float3(lightX, depth_of_field / 2, lightZ);
            lightDirection = float3.make_float3(0.0F, 0.0F, 0.0F) - lightLocation;
            MathFunctions.normalize(ref lightDirection);

            float viewRad = MathFunctions.toRad(viewAngle);
            float viewX = MathFunctions.cosf(viewRad) * depth_of_field / 2;
            float viewZ = MathFunctions.sinf(viewRad) * depth_of_field / 2;

            nearFieldLocation = float3.make_float3(viewX, 0.0F, viewZ);
            viewDirection = float3.make_float3(0.0F, 0.0F, 0.0F) - nearFieldLocation;
            MathFunctions.normalize(ref viewDirection);

            float3 reverseDirection = viewDirection * eyeDistanceFromNearField;
            MathFunctions.scalarMultiply(ref reverseDirection, eyeDistanceFromNearField);
            eyeLocation = nearFieldLocation - reverseDirection;
        }

        float[] quadVertices = {
            // Positions        // Texture Coordsg
            -1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
            1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };
        
        public Program() : base(1024, 1024, GraphicsMode.Default, "Hybridizer Mandelbulb", GameWindowFlags.Default)
        {
            WindowBorder = WindowBorder.Fixed; // disable resize
            Init();
            runner = HybRunner.Cuda().SetDistrib(32, 32, 16, 16, 1, 0);
            wrapped = runner.Wrap(new Mandelbulb());
            cuda.ERROR_CHECK(cuda.GetLastError());
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
        }

        protected int LoadShaderProgram(string VSSource, string FSSource)
        {
            int program = GL.CreateProgram();
            int vshader = GL.CreateShader(ShaderType.VertexShader);
            int fshader = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(vshader, VSSource);
            GL.ShaderSource(fshader, FSSource);
            GL.CompileShader(vshader);
            GL.CompileShader(fshader);
            int success;
            GL.GetShader(vshader, ShaderParameter.CompileStatus, out success);
            if (success != 1)
            {
                Console.WriteLine("vshader not compiled");
                Console.WriteLine(GL.GetShaderInfoLog(vshader));
            }

            GL.GetShader(fshader, ShaderParameter.CompileStatus, out success);
            if (success != 1)
            {
                Console.WriteLine("fshader not compiled");
                Console.WriteLine(GL.GetShaderInfoLog(fshader));
            }

            Console.WriteLine(GL.GetShaderInfoLog(vshader));
            Console.WriteLine(GL.GetShaderInfoLog(fshader));
            GL.AttachShader(program, vshader);
            GL.AttachShader(program, fshader);
            GL.LinkProgram(program);
            return program;
        }

        protected override void OnLoad(System.EventArgs e)
        {
            try
            {
                Console.WriteLine(ClientSize.Width + " " + ClientSize.Height);
                Console.WriteLine(GL.GetString(StringName.Version));
                Console.WriteLine(GL.GetString(StringName.ShadingLanguageVersion));
                GL.Viewport(0, 0, ClientSize.Width, ClientSize.Height);
                VSync = VSyncMode.On;
                GL.Enable(EnableCap.Texture2D);
                GL.Enable(EnableCap.DepthTest);
                GL.DepthFunc(DepthFunction.Less);

                // Setup quad VAO
                quadVAO = GL.GenVertexArray();
                int quadVBO = GL.GenBuffer();
                GL.BindVertexArray(quadVAO);
                GL.BindBuffer(BufferTarget.ArrayBuffer, quadVBO);
                GL.BufferData(BufferTarget.ArrayBuffer, quadVertices.Length * sizeof(float), quadVertices, BufferUsageHint.StaticDraw);
                GL.EnableVertexAttribArray(0);
                GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 5 * sizeof(float), 0);
                GL.EnableVertexAttribArray(1);
                GL.VertexAttribPointer(1, 2, VertexAttribPointerType.Float, false, 5 * sizeof(float), 3 * sizeof(float));

                uchar4[] data = new uchar4[Width * Height];

                textureID = GL.GenTexture();
                GL.BindTexture(TextureTarget.Texture2D, textureID);
                GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, Width, Height, 0, PixelFormat.Bgra, PixelType.UnsignedByte, data);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, Convert.ToInt32(TextureWrapMode.Repeat));
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, Convert.ToInt32(TextureWrapMode.Repeat));
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, Convert.ToInt32(TextureMinFilter.Linear));
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, Convert.ToInt32(TextureMagFilter.Linear));

                GL.BindTexture(TextureTarget.Texture2D, 0);

                string vsource = File.ReadAllText(@"vertex.glsl");
                string fsource = File.ReadAllText(@"fragment.glsl");
                shaderProgram = LoadShaderProgram(vsource, fsource);
                IntPtr resource;
                cuda.ERROR_CHECK(cuda.GraphicsGLRegisterImage(out resource, (uint)textureID, (uint)GL_TEXTURE_MODE.GL_TEXTURE_2D, (uint)cudaGraphicsRegisterFlags.SurfaceLoadStore));
                cuda.ERROR_CHECK(cuda.GraphicsMapResources(1, new IntPtr[1] { resource }, cudaStream_t.NO_STREAM));
                cudaArray_t array;
                cuda.ERROR_CHECK(cuda.GraphicsSubResourceGetMappedArray(out array, resource, 0, 0));


                cudaChannelFormatDesc channelDescSurf = TextureHelpers.cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKind.cudaChannelFormatKindUnsigned);
                cudaResourceDesc resDescSurf = TextureHelpers.CreateCudaResourceDesc(array);
                cuda.CreateSurfaceObject(out surface, ref resDescSurf);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }

        private static double averageFPS = 0.0;
        private static long elapsedMS = 0;
        private static int frameCounter = 0;
        private static Stopwatch stopwatch = new Stopwatch();
        private static double averageKernelDuration = 0;

        protected override void OnRenderFrame(FrameEventArgs e)
        {
            stopwatch.Start();
            viewAngle += 0.5F;
            lightAngle += 0.5F;
            Init();
            wrapped.Render(surface, Width, Height, iterations, viewDirection, nearFieldLocation, eyeLocation, lightDirection);
            cuda.ERROR_CHECK(cuda.GetLastError(), false);
            cuda.ERROR_CHECK(cuda.DeviceSynchronize(), false);
            GL.ClearColor(Color.Purple);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            GL.UseProgram(shaderProgram);
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, textureID);

            GL.BindVertexArray(quadVAO);
            GL.DrawArrays(PrimitiveType.TriangleStrip, 0, 4);
            GL.BindVertexArray(0);
            SwapBuffers();
            stopwatch.Stop();
            elapsedMS += stopwatch.ElapsedMilliseconds;
            stopwatch.Reset();
            averageFPS += RenderFrequency;
            frameCounter += 1;
            averageKernelDuration += runner.LastKernelDuration.ElapsedMilliseconds;
            if (elapsedMS > 1000)
            {
                averageFPS /= frameCounter;
                averageKernelDuration /= frameCounter;
                Console.Out.Write("\rKernel time : {0:N2} ms -- FPS : {1:N2}", averageKernelDuration, averageFPS);
                averageFPS = 0.0;
                elapsedMS = 0;
                frameCounter = 0;
                averageKernelDuration = 0;
            }
        }

        static void Main(string[] args)
        {
            try
            {
                using (Program p = new Program())
                {
                    p.Run();
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("[ERROR] : {0}", e.Message);
            }
        }
    }
}
