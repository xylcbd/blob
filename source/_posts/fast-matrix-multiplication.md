---
title: 快速矩阵乘法
date: 2016-07-19 20:09:04
categories:
 - 算法
 - 性能优化
tags:
 - gemm
 - 性能优化
---
## 矩阵乘法
矩阵乘法是高性能计算以及深度学习的基石之一，矩阵乘法的优化一直是相关业界的关注重点。  
矩阵乘法的定义非常简单，定义见：[wiki](https://en.wikipedia.org/wiki/Matrix_multiplication "matrix multiplication")。  

## 最简单的矩阵乘法实现
最简单的矩阵乘法实现，时间复杂度为O^3。
```c++
static void mm_generate(float* matA,float* matB,float* matC,const int M,const int N,const int K,const int strideA,const int strideB,const int strideC)
{
	for (int i = 0; i < M;i++)
	{
		for (int j = 0; j < N;j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < K;k++)
			{
				sum += matA[i*strideA + k] * matB[k*strideB + j];
			}
			matC[i*strideC + j] = sum;
		}
	}
}
```

## 转化为分块计算
矩阵转换为分块实现，时间复杂度仍然是O^3。
```c++
static void mm_split(float* matA, float* matB, float* matC, const int M, const int N, const int K, const int strideA, const int strideB, const int strideC)
{
	memset(matC, 0, M*strideC*sizeof(float));
	//C11 = A11xB11 + A12XB21
	for (int i = 0; i < M/2; i++)
	{
		for (int j = 0; j < N/2; j++)
		{
			float sum;
			//A11XB11
			sum = 0.0f;
			for (int k = 0; k < K/2; k++)
			{
				sum += matA[i*strideA + k] * matB[k*strideB + j];
			}
			matC[i*strideC + j] += sum;
			//A12XB21
			sum = 0.0f;
			for (int k = K / 2; k < K; k++)
			{
				sum += matA[i*strideA + k] * matB[k*strideB + j];
			}
			matC[i*strideC + j] += sum;
		}
	}
	//C12 = A11XB12 + A12XB22
	for (int i = 0; i < M / 2; i++)
	{
		for (int j = N/2; j < N; j++)
		{
			float sum;
			//A11XB11
			sum = 0.0f;
			for (int k = 0; k < K / 2; k++)
			{
				sum += matA[i*strideA + k] * matB[k*strideB + j];
			}
			matC[i*strideC + j] += sum;
			//A12XB21
			sum = 0.0f;
			for (int k = K / 2; k < K; k++)
			{
				sum += matA[i*strideA + k] * matB[k*strideB + j];
			}
			matC[i*strideC + j] += sum;
		}
	}
	//C21 = A21XB11 + A22XB21
	for (int i = M/2; i < M; i++)
	{
		for (int j = 0; j < N / 2; j++)
		{
			float sum;
			//A11XB11
			sum = 0.0f;
			for (int k = 0; k < K / 2; k++)
			{
				sum += matA[i*strideA + k] * matB[k*strideB + j];
			}
			matC[i*strideC + j] += sum;
			//A12XB21
			sum = 0.0f;
			for (int k = K / 2; k < K; k++)
			{
				sum += matA[i*strideA + k] * matB[k*strideB + j];
			}
			matC[i*strideC + j] += sum;
		}
	}
	//C22 = A21XB12 + A22XB22
	for (int i = M/2; i < M; i++)
	{
		for (int j = N/2; j < N; j++)
		{
			float sum;
			//A11XB11
			sum = 0.0f;
			for (int k = 0; k < K / 2; k++)
			{
				sum += matA[i*strideA + k] * matB[k*strideB + j];
			}
			matC[i*strideC + j] += sum;
			//A12XB21
			sum = 0.0f;
			for (int k = K / 2; k < K; k++)
			{
				sum += matA[i*strideA + k] * matB[k*strideB + j];
			}
			matC[i*strideC + j] += sum;
		}
	}
}
```

## Strassen算法
Strassen算法，时间复杂度是O^2.81。定义见：[wiki](https://en.wikipedia.org/wiki/Strassen_algorithm "Strassen algorithm")  
c++实现：  
```c++
static void mm_strassen(float* matA, float* matB, float* matC, const int M, const int N, const int K, const int strideA, const int strideB, const int strideC)
{
	if ((M <= 64) || (M%2 != 0 ||N%2 != 0 ||K%2!=0))
	{
		return mm_generate(matA, matB, matC, M, N, K, strideA, strideB, strideC);
	}
	memset(matC, 0, M*strideC*sizeof(float));
	int offset = 0;

	//M1 = (A11+A22)*(B11+B22)
	std::vector<float> M1((M / 2) * (N / 2));
	{
		memset(&M1[0], 0, M1.size()*sizeof(float));
		//M1_0 = (A11+A22)
		std::vector<float> M1_0((M / 2) * (K / 2));
		offset = M*strideA / 2 + K / 2;
		for (int i = 0; i < M / 2; i++)
		{
			for (int j = 0; j < K/2; j++)
			{
				const int baseIdx = i*strideA + j;
				M1_0[i*K/2+j] = matA[baseIdx] + matA[baseIdx + offset];
			}
		}
		//M1_1 = (B11+B22)
		std::vector<float> M1_1((K / 2) * (N / 2));
		offset = K*strideB / 2 + N / 2;
		for (int i = 0; i < K / 2; i++)
		{
			for (int j = 0; j < N / 2; j++)
			{
				const int baseIdx = i*strideB + j;
				M1_1[i*N/2+j] = matB[baseIdx] + matB[baseIdx + offset];
			}
		}
		mm_strassen(&M1_0[0], &M1_1[0], &M1[0], M / 2, N / 2, K / 2,
			K/2,N/2,N/2);
	}

	//M2 = (A21+A22)*B11
	std::vector<float> M2((M / 2) * (N / 2));
	{
		memset(&M2[0], 0, M2.size()*sizeof(float));
		//M2_0 = (A21+A22)
		std::vector<float> M2_0((M / 2) * (K / 2));
		offset = K / 2;
		for (int i = M / 2; i < M; i++)
		{
			for (int j = 0; j < K / 2; j++)
			{
				const int baseIdx = i*strideA + j;
				M2_0[(i-M/2)*K/2+j] = matA[baseIdx] + matA[baseIdx + offset];
			}
		}
		//M2_2 = B11
		mm_strassen(&M2_0[0], &matB[N / 2], &M2[0], M / 2, N / 2, K / 2,
			K / 2, strideB, N / 2);
	}

	//M3 = A11*(B12-B22)
	std::vector<float> M3((M / 2) * (N / 2));
	{
		memset(&M3[0], 0, M3.size()*sizeof(float));
		//M3_0 = A11
		//M3_1 = (B12-B22)
		std::vector<float> M3_1((K / 2) * (N / 2));
		offset = K*strideB / 2;
		for (int i = 0; i < K/2; i++)
		{
			for (int j = N/2; j < N; j++)
			{
				const int baseIdx = i*strideB + j;
				M3_1[i*N/2+j-N/2] = matB[baseIdx] - matB[baseIdx + offset];
			}
		}
		mm_strassen(matA, &M3_1[0], &M3[0], M / 2, N / 2, K / 2,
			strideA, N / 2, N / 2);
	}

	//M4 = A22*(B21-B11)
	std::vector<float> M4((M / 2) * (N / 2));
	{
		memset(&M4[0], 0, M4.size()*sizeof(float));
		//M4_0 = A22
		//M4_1 = (B12-B22)
		std::vector<float> M4_1((K / 2) * (N / 2));
		offset = K*strideB / 2;
		for (int i = 0; i < K / 2; i++)
		{
			for (int j = N / 2; j < N; j++)
			{
				const int baseIdx = i*strideB + j;
				M4_1[i*N/2+j-N/2] = matB[baseIdx + offset] - matB[baseIdx];
			}
		}
		mm_strassen(matA + M*strideA / 2 + K / 2, &M4_1[0], &M4[0], M / 2, K / 2, N / 2,
			strideA, N / 2, N / 2);
	}

	//M5 = (A11+A12)*B22
	std::vector<float> M5((M / 2) * (N / 2));
	{
		memset(&M5[0], 0, M5.size()*sizeof(float));
		//M5_0 = (A11+A12)
		std::vector<float> M5_0((M / 2) * (K / 2));
		offset = K / 2;
		for (int i = 0; i < M/2; i++)
		{
			for (int j = 0; j < K / 2; j++)
			{
				const int baseIdx = i*strideA + j;
				M5_0[i*K / 2 + j] = matA[baseIdx] + matA[baseIdx + offset];
			}
		}
		//M5_1 = B22
		mm_strassen(&M5_0[0], &matB[K*strideB / 2 + N / 2], &M5[0], M / 2, N / 2, K / 2,
			K / 2, strideB, N / 2);
	}

	//M6 = (A21-A11)*(B11+B12)
	std::vector<float> M6((M / 2) * (N / 2));
	{
		memset(&M6[0], 0, M6.size()*sizeof(float));
		//M6_0 = (A21-A11)
		std::vector<float> M6_0((M / 2) * (K / 2));
		offset = K*N / 2;
		for (int i = 0; i < M / 2; i++)
		{
			for (int j = 0; j < K/2; j++)
			{
				const int baseIdx = i*strideA + j;
				M6_0[i*K/2+j] = matA[baseIdx + offset] - matA[baseIdx];
			}
		}
		//M6_1 = (B11+B12)
		std::vector<float> M6_1((K / 2) * (N / 2));
		offset = N / 2;
		for (int i = 0; i < K / 2; i++)
		{
			for (int j = 0; j < N/2; j++)
			{
				const int baseIdx = i*strideB + j;
				M6_1[i*N/2+j] = matB[baseIdx] + matB[baseIdx + offset];
			}
		}
		mm_strassen(&M6_0[0], &M6_1[0], &M6[0], M / 2, N / 2, K / 2,
			K / 2, N / 2, N / 2);
	}

	//M7 = (A12-A22)*(B21+B22)
	std::vector<float> M7((M / 2) * (N / 2));
	{
		memset(&M7[0], 0, M7.size()*sizeof(float));
		//M7_0 = (A12-A22)
		std::vector<float> M7_0((M / 2) * (K / 2));
		offset = M*strideA / 2;
		for (int i = 0; i < M / 2; i++)
		{
			for (int j = K/2; j < K; j++)
			{
				const int baseIdx = i*strideA + j;
				M7_0[i*K / 2 + j - K / 2] = matA[baseIdx] - matA[baseIdx + offset];
			}
		}
		//M7_1 = (B21+B22)
		std::vector<float> M7_1((K / 2) * (N / 2));
		offset = N / 2;
		for (int i = K/2; i < K; i++)
		{
			for (int j = 0; j < N / 2; j++)
			{
				const int baseIdx = i*strideB + j;
				M7_1[(i-K/2)*N / 2 + j] = matB[baseIdx] + matB[baseIdx + offset];
			}
		}
		mm_strassen(&M7_0[0], &M7_1[0], &M7[0], M / 2, N / 2, K / 2,
			K / 2, N / 2, N / 2);
	}	
	for (int i = 0; i < M / 2;i++)
	{
		for (int j = 0; j < N / 2;j++)
		{
			const int idx = i*N / 2 + j;
			//C11 = M1+M4-M5+M7
			matC[i*strideC + j] = M1[idx] + M4[idx] - M5[idx] + M7[idx];
			//C12 = M3+M5
			matC[i*strideC + j + N/2] = M3[idx] + M5[idx];
			//C21 = M2+M4
			matC[(i+M/2)*strideC + j] = M2[idx] + M4[idx];
			//C22 = M1-M2+M3+M6
			matC[(i+M/2)*strideC + j + N/2] = M1[idx] - M2[idx] + M3[idx] + M6[idx];
		}
	}
}
```

## Coppersmith-Winograd算法
Coppersmith-Winograd算法，时间复杂度是O^2.38。定义见：[wiki](https://en.wikipedia.org/wiki/Coppersmith-Winograd_algorithm "Coppersmith-Winograd algorithm")  
c++实现：  
```c++
static void mm_winograd(float* matA, float* matB, float* matC, const int M, const int N, const int K, const int strideA, const int strideB, const int strideC)
{
	if ((M <= 64) || (M % 2 != 0 || N % 2 != 0 || K % 2 != 0))
	{
		return mm_generate(matA, matB, matC, M, N, K, strideA, strideB, strideC);
	}
	memset(matC, 0, M*strideC*sizeof(float));
	int offset = 0;

	std::vector<float> S1((M / 2) * (K / 2));
	std::vector<float> S2((M / 2) * (K / 2));
	std::vector<float> S3((M / 2) * (K / 2));
	std::vector<float> S4((M / 2) * (K / 2));
	for (int i = 0; i < M / 2;i++)
	{
		for (int j = 0; j < K / 2;j++)
		{
			const int idx = i*K / 2 + j;
			//S1 = A21 + A22
			S1[idx] = matA[(i + M / 2)*strideA + j] + matA[(i + M / 2)*strideA + j + K / 2];
			//S2 = S1 - A11
			S2[idx] = S1[idx] - matA[i*strideA + j];
			//S3 = A11 - A21
			S3[idx] = matA[i*strideA + j] - matA[(i + M / 2)*strideA + j];
			//S4 = A12 - S2
			S4[idx] = matA[i*strideA + j + K / 2] - S2[idx];
		}
	}
	std::vector<float> T1((K / 2) * (N / 2));
	std::vector<float> T2((K / 2) * (N / 2));
	std::vector<float> T3((K / 2) * (N / 2));	
	std::vector<float> T4((K / 2) * (N / 2));
	for (int i = 0; i < K / 2; i++)
	{
		for (int j = 0; j < N / 2; j++)
		{
			const int idx = i*N / 2 + j;
			//T1 = B21 - B11
			T1[idx] = matB[(i + K / 2)*strideB + j] - matB[i*strideB + j];
			//T2 = B22 - T1
			T2[idx] = matB[(i + K / 2)*strideB + j + N / 2] - T1[idx];
			//T3 = B22 - B12
			T3[idx] = matB[(i + K / 2)*strideB + j + N / 2] - matB[i*strideB + j + N / 2];
			//T4 = T2 - B21
			T4[idx] = T2[idx] - matB[(i + K / 2)*strideB + j];
		}
	}

	//M1 = A11*B11
	std::vector<float> M1((M / 2) * (N / 2));
	{
		memset(&M1[0], 0, M1.size()*sizeof(float));
		mm_winograd(matA, matB, &M1[0], M / 2, N / 2, K / 2,
			strideA, strideB, N / 2);
	}

	//M2 = A12*B21
	std::vector<float> M2((M / 2) * (N / 2));
	{
		memset(&M2[0], 0, M2.size()*sizeof(float));
		mm_winograd(matA + K / 2, matB + K*strideB/2, &M2[0], M / 2, N / 2, K / 2,
			strideA, strideB, N / 2);
	}

	//M3 = S4*B22
	std::vector<float> M3((M / 2) * (N / 2));
	{
		memset(&M3[0], 0, M3.size()*sizeof(float));
		mm_winograd(&S4[0], matB + K*strideB/2 + N / 2, &M3[0], M / 2, N / 2, K / 2,
			K/2, strideB, N / 2);
	}

	//M4 = A22*T4
	std::vector<float> M4((M / 2) * (N / 2));
	{
		memset(&M4[0], 0, M4.size()*sizeof(float));
		mm_winograd(matA + M*strideA / 2 + K / 2, &T4[0], &M4[0], M / 2, N / 2, K / 2,
			strideA, N / 2, N / 2);
	}

	//M5 = S1*T1
	std::vector<float> M5((M / 2) * (N / 2));
	{
		memset(&M5[0], 0, M5.size()*sizeof(float));		
		mm_winograd(&S1[0], &T1[0], &M5[0], M / 2, N / 2, K / 2,
			K / 2, N/2, N / 2);
	}

	//M6 = S2*T2
	std::vector<float> M6((M / 2) * (N / 2));
	{
		memset(&M6[0], 0, M6.size()*sizeof(float));
		mm_winograd(&S2[0], &T2[0], &M6[0], M / 2, N / 2, K / 2,
			K / 2, N / 2, N / 2);
	}

	//M7 = S3*T3
	std::vector<float> M7((M / 2) * (N / 2));
	{
		memset(&M7[0], 0, M7.size()*sizeof(float));		
		mm_winograd(&S3[0], &T3[0], &M7[0], M / 2, N / 2, K / 2,
			K / 2, N / 2, N / 2);
	}

	for (int i = 0; i < M / 2; i++)
	{
		for (int j = 0; j < N / 2; j++)
		{
			const int idx = i*N / 2 + j;
			//U1 = M1 + M2
			const auto U1 = M1[idx] + M2[idx];
			//U2 = M1 + M6
			const auto U2 = M1[idx] + M6[idx];
			//U3 = U2 + M7
			const auto U3 = U2 + M7[idx];
			//U4 = U2 + M5
			const auto U4 = U2 + M5[idx];
			//U5 = U4 + M3
			const auto U5 = U4 + M3[idx];
			//U6 = U3 - M4
			const auto U6 = U3 - M4[idx];
			//U7 = U3 + M5
			const auto U7 = U3 + M5[idx];

			//C11 = U1
			matC[i*strideC + j] = U1;
			//C12 = U5
			matC[i*strideC + j + N / 2] = U5;
			//C21 = U6
			matC[(i + M / 2)*strideC + j] = U6;
			//C22 = U7
			matC[(i + M / 2)*strideC + j + N / 2] = U7;
		}
	}
}
```

## 其他优化方法
除了在时间复杂度上的优化以外，还可以通过多线程、SIMD指令、GPGPU等工程方法进行优化。

## 参考
* http://www.cs.tau.ac.il/~zwick/Adv-Alg-2015/Matrix-Graph-Algorithms.pptx
