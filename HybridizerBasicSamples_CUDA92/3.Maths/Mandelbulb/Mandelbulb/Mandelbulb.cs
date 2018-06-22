using Hybridizer.Runtime.CUDAImports;
using System;

namespace Mandelbulb
{
    // inspired by https://github.com/royvanrijn/mandelbulb.js
    public class Mandelbulb
    {
        public int Iterations;
        public int MAX_ITER = 5000;
        public float DEPTH_OF_FIELD;

        public float Distance(float3 pos)
        {
            float3 z = pos;
            var dr = 1.0F;
            var r = 0.0F;
            for (var i = 0; i < Iterations; i++)
            {
                r = MathFunctions.length(z);
                if (r > DEPTH_OF_FIELD) break;

                var theta = MathFunctions.acosf(z.z / r);
                var phi = MathFunctions.atan2f(z.y, z.x);
                float r2 = r * r;
                float r4 = r2 * r2;
                float r7 = r * r2 * r4;
                float r8 = r7 * r;
                dr = r7 * 8.0F * dr + 1.0F;
                var zr = r8;
                theta = theta * 8.0F;
                phi = phi * 8.0F;
                float costheta, sintheta, cosphi, sinphi;
                MathFunctions.sincosf(theta, out sintheta, out costheta);
                MathFunctions.sincosf(phi, out sinphi, out cosphi);
                z.x = sintheta * cosphi;
                z.y = sinphi * sintheta;
                z.z = costheta;
                z = (z * zr) + pos;
            }

            return 0.5F * MathFunctions.logf(r) * r / dr;
        }
        
        public float shadow(float3 lightDirection, float3 rayLocation, float mint, float maxt, float k)
        {
            var res = 1.0F;
            for (float t = mint; t < maxt;)
            {
                float3 rd = lightDirection * t;
                float3 ro = rayLocation - rd;
                var h = Distance(ro);
                if (h < 0.001)
                {
                    return 0.0F;
                }

                res = Math.Min(res, k * h / t);
                t += h;
            }

            return res;
        }

        [EntryPoint]
        public void Render(uchar4[] imageData, int cWidth, int cHeight, float3 viewDirection, float3 nearFieldLocation, float3 eyeLocation, float3 lightDirection)
        {
            int cHalfWidth = cWidth / 2;
            float pixel = ((float) DEPTH_OF_FIELD) / (((cHeight + cWidth) / 2.0F));
            float halfPixel = pixel / 2.0F;
            float smallStep = 0.01F;
            float bigStep = 0.02F;
            for (int y = threadIdx.y + blockIdx.y * blockDim.y; y < cHeight; y += blockDim.y * gridDim.y)
            {
                var ny = y - cHeight / 2;

                float3 tempViewDirectionY = viewDirection;
                float3 tempViewDirectionX1 = viewDirection;
                MathFunctions.turnOrthogonal(ref tempViewDirectionY);
                MathFunctions.crossProduct(ref tempViewDirectionY, viewDirection);
                tempViewDirectionY = tempViewDirectionY * (ny * pixel);
                
                MathFunctions.turnOrthogonal(ref tempViewDirectionX1);

                for (int x = threadIdx.x + blockDim.x * blockIdx.x; x < cWidth; x += blockDim.x * gridDim.x)
                {
                    int nx = x - cHalfWidth;
                    float3 pixelLocation = nearFieldLocation;
                    float3 tempViewDirectionX2 = tempViewDirectionX1 * (nx * pixel);
                    pixelLocation = pixelLocation + tempViewDirectionX2 + tempViewDirectionY;

                    float3 rayLocation = pixelLocation;
                    float3 rayDirection = rayLocation - eyeLocation;
                    MathFunctions.normalize(ref rayDirection);

                    var distanceFromCamera = 0.0;
                    var d = Distance(rayLocation);

                    var iterations = 0;
                    for (; iterations < MAX_ITER; iterations++)
                    {

                        if (d < halfPixel)
                        {
                            break;
                        }

                        //Increase rayLocation with direction and d:
                        rayDirection = rayDirection * d;
                        rayLocation = rayLocation + rayDirection;
                        //And reset ray direction:
                        MathFunctions.normalize(ref rayDirection);

                        //Move the pixel location:
                        distanceFromCamera = MathFunctions.length(nearFieldLocation - rayLocation);

                        if (distanceFromCamera > DEPTH_OF_FIELD)
                        {
                            break;
                        }

                        d = Distance(rayLocation);
                    }

                    if (distanceFromCamera < DEPTH_OF_FIELD && distanceFromCamera > 0)
                    {

                        rayLocation.x -= smallStep;
                        var locationMinX = Distance(rayLocation);
                        rayLocation.x += bigStep;
                        var locationPlusX = Distance(rayLocation);
                        rayLocation.x -= smallStep;

                        rayLocation.y -= smallStep;
                        var locationMinY = Distance(rayLocation);
                        rayLocation.y += bigStep;
                        var locationPlusY = Distance(rayLocation);
                        rayLocation.y -= smallStep;

                        rayLocation.z -= smallStep;
                        var locationMinZ = Distance(rayLocation);
                        rayLocation.z += bigStep;
                        var locationPlusZ = Distance(rayLocation);
                        rayLocation.z -= smallStep;

                        float3 normal = new float3();
                        //Calculate the normal:
                        normal.x = (locationMinX - locationPlusX);
                        normal.y = (locationMinY - locationPlusY);
                        normal.z = (locationMinZ - locationPlusZ);
                        MathFunctions.normalize(ref normal);

                        //Calculate the ambient light:
                        var dotNL = MathFunctions.dotProduct(lightDirection, normal);
                        var diff = MathFunctions.saturate(dotNL);

                        //Calculate specular light:
                        float3 halfway = rayDirection + lightDirection;
                        MathFunctions.normalize(ref halfway);

                        var dotNH = MathFunctions.dotProduct(halfway, normal);
                        float s = MathFunctions.saturate(dotNH);
                        float s2 = s * s;
                        float s4 = s2 * s2;
                        float s8 = s4 * s4;
                        float s16 = s8 * s8;
                        float s32 = s16 * s16;
                        float spec = s * s2 * s32; // s^35

                        var shad = shadow(lightDirection, rayLocation, 1.0F, DEPTH_OF_FIELD, 16.0F) + 0.25F;
                        var brightness = (10.0F + (200.0F + spec * 45.0F) * shad * diff) / 270.0F;

                        var red = 10 + (380 * brightness);
                        var green = 10 + (280 * brightness);
                        var blue = (180 * brightness);

                        red = MathFunctions.clamp(red, 0, 255.0F);
                        green = MathFunctions.clamp(green, 0, 255.0F);
                        blue = MathFunctions.clamp(blue, 0, 255.0F);

                        var pixels = ((y * cWidth) + x);
                        imageData[pixels].x = (byte) red;
                        imageData[pixels].y = (byte) green;
                        imageData[pixels].z = (byte) blue;
                    }
                    else
                    {
                        var pixels = ((y * cWidth) + x);
                        imageData[pixels].x = (byte) (155.0F + MathFunctions.clamp(iterations * 1.5F, 0.0F, 100.0F));
                        imageData[pixels].y =  (byte) (205.0F + MathFunctions.clamp(iterations * 1.5F, 0.0F, 50.0F));
                        imageData[pixels].z = 255;
                    }
                }
            }
        }
    }
}
