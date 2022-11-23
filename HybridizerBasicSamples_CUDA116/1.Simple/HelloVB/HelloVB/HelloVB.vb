Imports Hybridizer.Runtime.CUDAImports

<Assembly: HybRunnerDefaultSatelliteName("HelloVB_CUDA.dll")>

Class HelloVB

    <EntryPoint>
    Shared Sub Add(ByVal a As Single(), ByVal b As Single(), ByVal N As Integer)
        For i As Integer = threadIdx.x + blockIdx.x * blockDim.x To N - 1 Step gridDim.x * blockDim.x
            a(i) += b(i)
        Next
    End Sub

    Shared Sub Main()
        Dim N As Integer = 1024 * 1024 * 32
        Dim a(N - 1) As Single
        Dim b(N - 1) As Single

        For i As Integer = 0 To N - 1
            a(i) = 1.0F
            b(i) = i
        Next

        Dim wrapped As Object = HybRunner.Cuda().Wrap(New HelloVB())

        wrapped.Add(a, b, N)

        cuda.DeviceSynchronize()
        Console.Out.WriteLine(String.Join(", ", a.Take(35)))
    End Sub
End Class
