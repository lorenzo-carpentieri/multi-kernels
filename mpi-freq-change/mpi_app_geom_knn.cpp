#include <iostream>
#include <sycl/sycl.hpp>
#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <synergy.hpp>
#include <chrono>
using namespace sycl;

#define NUM_GEOM_RUN 3
#define NUM_KNN_RUN 1
#define NUM_ITERS_GEOM 400
#define NUM_ITERS_KNN 3

// Size of the vector
constexpr size_t N = 2048;

// generate random floating point numbers in the range [0,1]
float randomFloat()
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void fillrandom_float(float *arrayPtr, int width, int height, float rangeMin, float rangeMax)
{
    if (!arrayPtr)
    {
        fprintf(stderr, "Cannot fill array: NULL pointer.\n");
        return;
    }
    srand(7);
    double range = (double)(rangeMax - rangeMin);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            int index = i * width + j;
            arrayPtr[index] = rangeMin + (float)(range * ((float)rand() / (float)RAND_MAX));
        }
}

// init geom vector
void init_geom(std::vector<sycl::float16> &input,
               std::vector<float> &output)
    
{
    fillrandom_float((float*)input.data(), output.size(), 16, 0.001f, 100000.f);
}

// init knn vector
void init_knn(std::vector<sycl::float4> &input, std::vector<sycl::float4> &output)
{
    for (int i = 0; i < N; ++i)
    {
        input[i] = randomFloat();
        output[i] = 0;
    }
}

// class geometric_mean
class geometric_mean
{
private:
    int size;
    int num_iters;
    int chunk_size;
    const sycl::accessor<sycl::float16, 1, sycl::access_mode::read> in;
    sycl::accessor<float, 1, sycl::access_mode::read_write> out;

public:
    geometric_mean(int size,
                   int num_iters,
                   int chunk_size,
                   const sycl::accessor<sycl::float16, 1, sycl::access_mode::read> in,
                   sycl::accessor<float, 1, sycl::access_mode::read_write> out)
        : size(size), num_iters(num_iters), chunk_size(chunk_size), in(in), out(out) {}

    void operator()(sycl::id<1> id) const
    {
        int gid = id[0];
        if (gid >= size)
            return;

        for (size_t i = 0; i < num_iters; i++)
        {
            sycl::float16 val = in[gid];

            float mean = sycl::log(val.s0()) + sycl::log(val.s1()) + sycl::log(val.s2()) + sycl::log(val.s3()) +
                         sycl::log(val.s4()) + sycl::log(val.s5()) + sycl::log(val.s6()) + sycl::log(val.s7()) +
                         sycl::log(val.s8()) + sycl::log(val.s9()) + sycl::log(val.sA()) + sycl::log(val.sB()) +
                         sycl::log(val.sC()) + sycl::log(val.sD()) + sycl::log(val.sE()) + sycl::log(val.sF());
            mean /= chunk_size;

            float euler = 2.718281828459045235f;

            out[gid] = sycl::pow(euler, mean);
        }
    }
};

// knn
class knn
{
private:
    int size;
    int num_iters;
    int nRef;
    const sycl::accessor<float, 1, sycl::access_mode::read> ref_acc;
    const sycl::accessor<float, 1, sycl::access_mode::read> query_acc;
    const sycl::accessor<float, 1, sycl::access_mode::write> dist_acc;
    const sycl::accessor<int, 1, sycl::access_mode::write> neighbours_acc;

public:
    knn(int size, int num_iters, int nRef,
        const sycl::accessor<float, 1, sycl::access_mode::read> ref_acc,
        const sycl::accessor<float, 1, sycl::access_mode::read> query_acc,
        const sycl::accessor<float, 1, sycl::access_mode::write> dist_acc,
        const sycl::accessor<int, 1, sycl::access_mode::write> neighbours_acc)
        : size(size),
          num_iters(num_iters),
          nRef(nRef), ref_acc(ref_acc), query_acc(query_acc), dist_acc(dist_acc), neighbours_acc(neighbours_acc) {}

    void operator()(sycl::id<1> id) const
    {
        size_t gid = id[0];

        if (gid >= size)
            return;

        for (size_t i = 0; i < num_iters; i++)
        {
            size_t queryOffset = gid /* dim*/;

            size_t curNeighbour = 0;
            float curDist = MAXFLOAT;

            for (int i = 0; i < nRef; ++i)
            {
                float privateDist = 0;
                size_t refOffset = i /* dim*/;

#if F4
                float4 tmpDist = {0, 0, 0, 0};
                int d;
                for (d = 0; d < dim - 3; d += 4)
                {
                    /* Cypress code */
                    float4 a = {ref[refOffset + d], ref[refOffset + d + 1], ref[refOffset + d + 2], ref[refOffset + d + 3]};
                    float4 b = {query[queryOffset + d], query[queryOffset + d + 1], query[queryOffset + d + 2],
                                query[queryOffset + d + 3]};
                    float4 t = a - b; //(float4){ref[refOffset + d], ref[refOffset + d+1], ref[refOffset + d+2],
                                      // ref[refOffset + d+3]} - (float4){query[queryOffset + d], query[queryOffset + d+1],
                                      // query[queryOffset + d+2], query[queryOffset + d+3]};
                    //            a = t * t;
                    tmpDist += t * t;
                    //            privateDist = a.s0 + a.s1 + a.s2 + a.s3;*/
                }

                for (; d < dim; d++)
                {
                    float t = ref[refOffset + d] - query[queryOffset + d];
                    privateDist += t * t;
                }
                privateDist = tmpDist.s0 + tmpDist.s1 + tmpDist.s2 + tmpDist.s3;
#else
                //      for(int d = 0; d < dim; d++)
                {
                    float t = ref_acc[refOffset /* + d*/] - query_acc[queryOffset /* + d*/];
                    privateDist += t * t;
                }
#endif

                if (privateDist < curDist)
                {
                    curDist = privateDist;
                    curNeighbour = i;
                }
            }

            dist_acc[gid] = sycl::sqrt(curDist);
            neighbours_acc[gid] = curNeighbour;
        }
    }
};

// Print the vector
void print_vector(float *data, size_t size, size_t rank)
{
    std::cout << "Process " << rank << ": [" << data[N - 1] << "]" << std::endl;
}

int main(int argc, char *argv[])
{
    int num_processes, rank;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const synergy::frequency GEOM_FREQ = 250;
    const synergy::frequency KNN_FREQ = 1400;

    // Initialize SYCL queue for each process
    const auto &devices = sycl::device::get_devices();
    if (devices.empty())
    {
        std::cerr << "No SYCL devices available" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<device> gpu_devices;

    for (size_t i = 0; i < devices.size(); i++)
    {
        if (devices[i].has(sycl::aspect::gpu) && devices[i].get_platform().get_info<sycl::info::platform::name>().find("OpenCL") == std::string::npos)
            gpu_devices.push_back(devices[i]);
    }
    // create vector for sobel and merse for each process
    // knn vector
    const size_t knn_size = 8192;
    std::vector<float> merse_ma(knn_size);
    std::vector<float> merse_b(knn_size);
    std::vector<float> merse_c(knn_size);
    std::vector<int> merse_seed(knn_size);

    // sobel
    const size_t geom_size = 16384;
    std::vector<sycl::float16> geom_input(geom_size);
    std::vector<float> geom_output(geom_size);

    // init_soble only rank_0
    // init_merse only rank_0
    // Broadcast to all process

    // The rank 0 process send the data to all other process
    if (rank == 0)
    {
        // init_merse(merse_ma, merse_b, merse_c, merse_seed);
        init_geom(geom_input, geom_output);
        // std::fill(input_data.begin(), input_data.end(), 1);
        // std::fill(result_data.begin(), result_data.end(), 0);
    }

    // Create SYnergy queue
    auto mpi_queue = synergy::queue(gpu_devices[rank]);
// sycl event that will be associated to the frequency change kernel
#if HIDING == 1
    sycl::event freq_change_event;
#endif
    // Time profiling
    auto start = std::chrono::high_resolution_clock::now();

// MPI_IBcast implementation
#ifdef WITH_MPI_ASYNCH
    if (rank == 0)
        std::cout << "Process: " << rank << " MPI_ASYNCH" << std::endl;
    // Broadcasts with MPI_Ibcast
    MPI_Request request[5];
    MPI_Ibcast(merse_ma.data(), geom_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Ibcast(merse_b.data(), geom_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Ibcast(merse_c.data(), geom_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD, &request[2]);
    MPI_Ibcast(merse_seed.data(), geom_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD, &request[3]);
    MPI_Ibcast(geom_input.data(), geom_size * sizeof(sycl::float16), MPI_BYTE, 0, MPI_COMM_WORLD, &request[4]);
// Hiding frequency change in MPI_Ibcast
#if HIDING == 1
    if (rank == 0)
        std::cout << "Process: " << rank << " MPI_HIDING" << std::endl;
    freq_change_event = mpi_queue.submit(0, GEOM_FREQ, [&](sycl::handler &cgh)
                                         { cgh.single_task([=]()
                                                           {
                                                               // Do nothing
                                                           }); }); // Set frequency
#endif
    MPI_Waitall(5, request, MPI_STATUSES_IGNORE);
#if HIDING == 1
    freq_change_event.wait();
#endif
// MPI_Bcast implementation
#else
    if (rank == 0)
        std::cout << "Process: " << rank << " MPI_SYNCH" << std::endl;
#if HIDING == 1
    if (rank == 0)
        std::cout << "Process: " << rank << " HIDING" << std::endl;
    freq_change_event = mpi_queue.submit(0, GEOM_FREQ, [&](sycl::handler &cgh)
                                         { cgh.single_task([=]()
                                                           {
                                                               // Do nothing
                                                           }); });
#endif
    // TODO: Having more process for frequency change
    //  Wait for all broadcasts to complete
    // MPI_Bcast(merse_ma.data(), merse_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    // MPI_Bcast(merse_b.data(), merse_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    // MPI_Bcast(merse_c.data(), merse_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    // MPI_Bcast(merse_seed.data(), merse_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    // Bcast for sobel
    MPI_Bcast(geom_input.data(), geom_size * sizeof(sycl::float16), MPI_BYTE, 0, MPI_COMM_WORLD);
// Waiting the sycl kernel for the frequency change
#if HIDING == 1
    freq_change_event.wait();
#endif
#endif
    if (rank == 0)
        std::cout << "Starting geom ..." << std::endl;

    {
        #if HIDING == 0
            if (rank == 0)
                std::cout << "Process: " << rank << " NO_HIDING" << std::endl;
            mpi_queue.submit(0, GEOM_FREQ, [&](sycl::handler &cgh)
                             { cgh.single_task([=]()
                                               {
                                                   // Do nothing
                                               }); })
                .wait(); // Set frequency
        #endif
        buffer<sycl::float16, 1> geom_input_buffer(geom_input.data(), range<1>(geom_size));
        buffer<float, 1> geom_output_buffer(geom_output.data(), range<1>(geom_size));

        for (int i = 0; i < NUM_GEOM_RUN; i++)
        {

            // launch sobel kernel
            mpi_queue.submit([&](sycl::handler &cgh)
                             {
                const sycl::accessor<sycl::float16, 1, sycl::access_mode::read> input_accessor = {geom_input_buffer,cgh};
                 sycl::accessor<float, 1, sycl::access_mode::read_write> output_accessor = {geom_output_buffer,cgh};
                cgh.parallel_for(range<1>(geom_size), geometric_mean(geom_size, NUM_ITERS_GEOM,16, input_accessor, output_accessor)); })
                .wait();
        }
    }

    std::vector<float> geom_reduced_data(geom_size);
  
#ifdef WITH_MPI_ASYNCH
    MPI_Request reduce_request;
    MPI_Ireduce(geom_output.data(), geom_reduced_data.data(), geom_size,
                MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD, &reduce_request);
#if HIDING == 1
    mpi_queue.submit(0, KNN_FREQ, [&](sycl::handler &cgh)
                     { cgh.single_task([=]()
                                       {
                                           // Do nothing
                                       }); })
        .wait(); // Set frequency
#endif
    // Wait for the reduction to complete
    MPI_Wait(&reduce_request, MPI_STATUS_IGNORE);
#else
// TODO: Add freq change with more process
#if HIDING == 1
    freq_change_event = mpi_queue.submit(0, KNN_FREQ, [&](sycl::handler &cgh)
                                         { cgh.single_task([=]()
                                                           {
                                                               // Do nothing
                                                           }); });
#endif
    // Reduce on sobel output
    MPI_Reduce(geom_output.data(), geom_reduced_data.data(), geom_size, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
// Waiting for the frequency change kernel
#if HIDING == 1
    freq_change_event.wait();
#endif

#endif

//     {
//         if (rank == 0)
//             std::cout << "Starting merse ..." << std::endl;

//         // TODO: Add frequency change when you run the kernel for comparing with the MPI_apporach
//         buffer<uint, 1> merse_ma_buffer(merse_ma.data(), range<1>(merse_size));
//         buffer<uint, 1> merse_b_buffer(merse_b.data(), range<1>(merse_size));
//         buffer<uint, 1> merse_c_buffer(merse_c.data(), range<1>(merse_size));
//         buffer<uint, 1> merse_seed_buffer(merse_seed.data(), range<1>(merse_size));
//         buffer<sycl::float4, 1> merse_result_buffer(merse_result.data(), range<1>(merse_size));

//         for (int i = 0; i < NUM_MERSE_RUN; i++)
//         {
// #if HIDING == 0
//             mpi_queue.submit(0, MERSE_FREQ, [&](sycl::handler &cgh)
//                              { cgh.single_task([=]()
//                                                {
//                                                    // Do nothing
//                                                }); })
//                 .wait(); // Set frequency
// #endif
//             // launch sobel kernel
//             mpi_queue.submit([&](sycl::handler &cgh)
//                              {
//                 sycl::accessor<sycl::float4, 1, sycl::access_mode::read_write> result_accessor = {merse_result_buffer,cgh};
//                 const sycl::accessor<uint, 1, sycl::access_mode::read> ma_accessor = {merse_ma_buffer,cgh};
//                 const sycl::accessor<uint, 1, sycl::access_mode::read> b_accessor = {merse_b_buffer,cgh};
//                 const sycl::accessor<uint, 1, sycl::access_mode::read> c_accessor = {merse_c_buffer,cgh};
//                 const sycl::accessor<uint, 1, sycl::access_mode::read> seed_accessor = {merse_seed_buffer,cgh};


//                 cgh.parallel_for(range<1>(merse_size), merse_twister(merse_size, NUM_ITERS_MERSE, ma_accessor, b_accessor, c_accessor, seed_accessor, result_accessor)); })
//                 .wait();
//         }
//     }
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_time_app = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Process: " << rank << ", time [ms]: " << elapsed_time_app.count() << std::endl;
#ifdef SYNERGY_DEVICE_PROFILING
    std::cout << "Process: " << rank << ", device energy: " << mpi_queue.device_energy_consumption() << " j\n";
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    // TODO: run the code for check
    // TODO: add frequency change with different approach (using a thread, launching a process that handle the freq.change)
    // TODO: add MPI_IReduce and MPI_IBcast

    return 0;
}