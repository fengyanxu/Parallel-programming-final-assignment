#include <iostream>
#include <cstdlib> // For rand function
#include <ctime>   // For clock function on non-Windows platforms
#include <xmmintrin.h> // SSE header

#ifdef _WIN32
#include <Windows.h>
#else
#include <ctime>
#endif

const int N = 2400; // Define the size of the matrix

float m[N][N]; // Define the matrix

void m_reset()
{
    for (int i = 0; i < N; i++)
    {
        m[i][i] = 1.0; // Set diagonal elements to 1.0
        for (int j = i + 1; j < N; j++)
            m[i][j] = rand() / (float)RAND_MAX; // Generate random values for upper triangular part
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] += m[k][j]; // Perform addition operation
}

void print_matrix()
{
    std::cout << "Generated Matrix:" << std::endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << m[i][j] << "\t"; // Print matrix elements
        }
        std::cout << std::endl;
    }
}

void gaussian_elimination_division_serial(float *A, int n)
{
#ifdef _WIN32
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
#else
    std::clock_t start = std::clock();
#endif

    for (int k = 0; k < n; ++k)
    {
        float diagonal = A[k * n + k]; // Load diagonal element
        for (int j = k + 1; j < n; ++j)
        {
            A[k * n + j] /= diagonal; // Serial division
        }

        // Non-vectorized elimination
        for (int i = k + 1; i < n; ++i)
        {
            float factor = A[i * n + k] / A[k * n + k]; // Compute factor
            for (int j = k + 1; j < n; ++j)
            {
                A[i * n + j] -= factor * A[k * n + j]; // Serial elimination
            }
            A[i * n + k] = 0.0; // Set A[i][k] to zero
        }
    }

#ifdef _WIN32
    QueryPerformanceCounter(&end);
    double interval = double(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    std::cout << "Elapsed time for Gaussian elimination with division: " << interval << " seconds" << std::endl;
#else
    std::clock_t end = std::clock();
    double interval = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time for Gaussian elimination with division: " << interval << " seconds" << std::endl;
#endif
}

void gaussian_elimination_division(float *A, int n)
{
#ifdef _WIN32
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
#else
    std::clock_t start = std::clock();
#endif

    for (int k = 0; k < n; ++k)
    {
        __m128 vt = _mm_set1_ps(A[k * n + k]); // Load diagonal element
        for (int j = k + 1; j < n; j += 4)     // 4-way vectorization with SSE
        {
            __m128 va = _mm_loadu_ps(&A[k * n + j]); // Load 4 consecutive elements
            va = _mm_div_ps(va, vt);                 // Vectorized division
            _mm_storeu_ps(&A[k * n + j], va);        // Store the result back to memory
        }

        // Non-vectorized elimination
        for (int i = k + 1; i < n; ++i)
        {
            float factor = A[i * n + k] / A[k * n + k];
            for (int j = k + 1; j < n; ++j)
            {
                A[i * n + j] -= factor * A[k * n + j];
            }
            A[i * n + k] = 0.0; // Set A[i][k] to zero
        }
    }

#ifdef _WIN32
    QueryPerformanceCounter(&end);
    double interval = double(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    std::cout << "Elapsed time for Gaussian elimination with division and vectorization: " << interval << " seconds" << std::endl;
#else
    std::clock_t end = std::clock();
    double interval = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time for Gaussian elimination with division and vectorization: " << interval << " seconds" << std::endl;
#endif
}

void gaussian_elimination_elimination(float *A, int n)
{
#ifdef _WIN32
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
#else
    std::clock_t start = std::clock();
#endif

    for (int k = 0; k < n; ++k)
    {
        // Non-vectorized division
        float diagonal = A[k * n + k];
        for (int j = k + 1; j < n; ++j)
        {
            A[k * n + j] /= diagonal;
        }

        // Vectorized elimination
        for (int i = k + 1; i < n; ++i)
        {
            __m128 vaik = _mm_set1_ps(A[i * n + k]); // Load element A[i][k]
            for (int j = k + 1; j < n; j += 4)         // 4-way vectorization with SSE
            {
                __m128 vakj = _mm_loadu_ps(&A[k * n + j]); // Load 4 consecutive elements from A[k][j]
                __m128 vaij = _mm_loadu_ps(&A[i * n + j]); // Load 4 consecutive elements from A[i][j]
                __m128 vx = _mm_mul_ps(vakj, vaik);        // Vectorized multiplication
                vaij = _mm_sub_ps(vaij, vx);               // Vectorized subtraction
                _mm_storeu_ps(&A[i * n + j], vaij);        // Store the result back to memory
            }
            A[i * n + k] = 0.0; // Set A[i][k] to zero
        }
    }

#ifdef _WIN32
    QueryPerformanceCounter(&end);
    double interval = double(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    std::cout << "Elapsed time for Gaussian elimination with vectorization: " << interval << " seconds" << std::endl;
#else
    std::clock_t end = std::clock();
    double interval = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time for Gaussian elimination with vectorization: " << interval << " seconds" << std::endl;
#endif
}

int main()
{
    m_reset(); // Reset the matrix
    //print_matrix(); // Print the generated matrix as test case

    const int n = N;
    float *A = new float[n * n];

    // Convert matrix m to 1D array A
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A[i * n + j] = m[i][j];
        }
    }

    // Perform Gaussian elimination and measure time
    gaussian_elimination_division_serial(A, n);
    gaussian_elimination_division(A, n);
    gaussian_elimination_elimination(A, n);
    system("pause");
    delete[] A; // Free dynamically allocated memory
    return 0;
}
