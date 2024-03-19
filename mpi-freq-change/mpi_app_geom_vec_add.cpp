#include <iostream>
#include <sycl/sycl.hpp>
#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <synergy.hpp>
#include <chrono>
using namespace sycl;

#define NUM_GEOM_RUN 100
#define NUM_VEC_ADD_RUNS 1
#define NUM_ITERS_GEOM 400
#define NUM_ITERS_VEC_ADD 400


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
    fillrandom_float((float *)input.data(), output.size(), 16, 0.001f, 100000.f);
}

// init knn vector
void init_vec_add(std::vector<int> &in1, std::vector<int> &in2)
{
    std::fill(in1.begin(), in1.end(), 1);
    std::fill(in2.begin(), in2.end(), 2);
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
class vec_add
{
private:
    int size;
    int num_iters;
    int nRef;
    const sycl::accessor<int, 1, sycl::access_mode::read> in1;
    const sycl::accessor<int, 1, sycl::access_mode::read> in2;
    const sycl::accessor<int, 1, sycl::access_mode::write> out;

public:
    vec_add(int size, int num_iters,
        const sycl::accessor<int, 1, sycl::access_mode::read> in1,
        const sycl::accessor<int, 1, sycl::access_mode::read> in2,
        const sycl::accessor<int, 1, sycl::access_mode::write> out)
        : size(size),
          num_iters(num_iters),
          in1(in1), in2(in2), out(out){}

    void operator()(sycl::id<1> gid) const
    {
       for(size_t i = 0; i < num_iters; i++) {
          out[gid] = in1[gid] + in2[gid];
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

    if (rank==0){
         // Benchmark info print
        #if WITH_MPI_ASYNCH == 1
            #if HIDING == 1
                std::cout<< "****** Test MPI_ASYNCH + HIDING******" <<std::endl;
            #else
                std::cout<< "****** Test MPI_ASYNCH + NO_HIDING******" <<std::endl;
            #endif
        #else
            #if HIDING == 1
                std::cout<< "****** Test MPI_SYNCH + HIDING******" <<std::endl;
            #else
                std::cout<< "****** Test MPI_SYNCH + NO_HIDING******" <<std::endl;
            #endif
            
        #endif
    }

    // Freq. for minimizin energy consumption on Intel Max 1100 GPU for geometric_mean and knn kernels
    const synergy::frequency GEOM_FREQ = 250;
    const synergy::frequency VEC_ADD_FREQ = 1150;

    // Initialize SYCL queue for each process
    const auto &devices = sycl::device::get_devices();
    if (devices.empty())
    {
        std::cerr << "No SYCL devices available" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Select only GPUs with LevelZero backend
    std::vector<device> gpu_devices;
    for (size_t i = 0; i < devices.size(); i++)
    {
        if (devices[i].has(sycl::aspect::gpu) && devices[i].get_platform().get_info<sycl::info::platform::name>().find("OpenCL") == std::string::npos)
            gpu_devices.push_back(devices[i]);
    }
    
    // NOTE: do not change the size the selected frequency is optimized for this input size 
    const size_t vec_add_size = 1048576;
    // knn vector data
    std::vector<int> vec_add_in1(vec_add_size);
    std::vector<int> vec_add_in2(vec_add_size);
    std::vector<int> vec_add_out(vec_add_size);

    // NOTE: do not change the size the selected frequency is optimized for this input size 
    const size_t geom_size = 16384;
    // geometric_mean vector data
    std::vector<sycl::float16> geom_input(geom_size);
    std::vector<float> geom_output(geom_size);

   
    // The rank 0 process send the data to all other process
    if (rank == 0)
    {
        init_vec_add(vec_add_in1, vec_add_in2);
        init_geom(geom_input, geom_output);
    }

    // Create SYnergy queue
    auto mpi_queue = synergy::queue(gpu_devices[rank]);

// sycl event that will be associated to the frequency change kernel
#if HIDING == 1
    sycl::event freq_change_event;
#endif
    // Time profiling
    auto start = std::chrono::high_resolution_clock::now();

    if (rank == 0)
        std::cout << "Starting geom ..." << std::endl;
        
    {

        buffer<sycl::float16, 1> geom_input_buffer(geom_input.data(), range<1>(geom_size));
        buffer<float, 1> geom_output_buffer(geom_output.data(), range<1>(geom_size));

        // PHASE_1 kernel geometric_mean
        for (int i = 0; i < NUM_GEOM_RUN; i++)
        {
// MPI_IBcast approach
#if WITH_MPI_ASYNCH == 1
            // Broadcasts with MPI_Ibcast
            MPI_Request request[3];
            MPI_Ibcast(vec_add_in1.data(), vec_add_size, MPI_INT, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Ibcast(vec_add_in2.data(), vec_add_size, MPI_INT, 0, MPI_COMM_WORLD, &request[1]);
            MPI_Ibcast(geom_input.data(), geom_size * sizeof(sycl::float16), MPI_BYTE, 0, MPI_COMM_WORLD, &request[2]);
// Hiding frequency change in MPI_Ibcast
#if HIDING == 1
            // if (rank == 0)
            //     std::cout << "Process: " << rank << " MPI_HIDING" << std::endl;
            freq_change_event = mpi_queue.submit(0, GEOM_FREQ, [&](sycl::handler &cgh)
                                                 { cgh.single_task([=]()
                                                                   {
                                                                       // Do nothing
                                                                   }); }); // Set frequency
#endif
            MPI_Waitall(3, request, MPI_STATUSES_IGNORE);
#if HIDING == 1
            freq_change_event.wait();
#endif
// MPI_Bcast implementation
#else
           
#if HIDING == 1
            // if (rank == 0)
            //     std::cout << "Process: " << rank << " HIDING" << std::endl;
            freq_change_event = mpi_queue.submit(0, GEOM_FREQ, [&](sycl::handler &cgh)
                                                 { cgh.single_task([=]()
                                                                   {
                                                                       // Do nothing
                                                                   }); });
#endif
            // TODO: Having more process for frequency change
            //  Wait for all broadcasts to complete
            MPI_Bcast(vec_add_in1.data(), vec_add_size, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(vec_add_in2.data(), vec_add_size, MPI_INT, 0, MPI_COMM_WORLD);
            // Bcast for geom
            MPI_Bcast(geom_input.data(), geom_size * sizeof(sycl::float16), MPI_BYTE, 0, MPI_COMM_WORLD);
// Waiting the sycl kernel for the frequency change
#if HIDING == 1
            freq_change_event.wait();
#endif
#endif
            #if HIDING == 1
                // launch geometric_mean kernel
                mpi_queue.submit([&](sycl::handler &cgh)
                                {
                    const sycl::accessor<sycl::float16, 1, sycl::access_mode::read> input_accessor = {geom_input_buffer,cgh};
                    sycl::accessor<float, 1, sycl::access_mode::read_write> output_accessor = {geom_output_buffer,cgh};
                    cgh.parallel_for(range<1>(geom_size), geometric_mean(geom_size, NUM_ITERS_GEOM,16, input_accessor, output_accessor)); })
                    .wait();
            #else 
                // launch geometric_mean kernel
                mpi_queue.submit(0, GEOM_FREQ, [&](sycl::handler &cgh)
                                {
                    const sycl::accessor<sycl::float16, 1, sycl::access_mode::read> input_accessor = {geom_input_buffer,cgh};
                    sycl::accessor<float, 1, sycl::access_mode::read_write> output_accessor = {geom_output_buffer,cgh};
                    cgh.parallel_for(range<1>(geom_size), geometric_mean(geom_size, NUM_ITERS_GEOM,16, input_accessor, output_accessor)); })
                    .wait();
            #endif
        }
    }

    std::vector<float> geom_reduced_data(geom_size);

    #if WITH_MPI_ASYNCH == 1
      MPI_Request reduce_request;
      MPI_Ireduce(geom_output.data(), geom_reduced_data.data(), geom_size,
                  MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD, &reduce_request);
      #if HIDING == 1 
          freq_change_event = mpi_queue.submit(0, VEC_ADD_FREQ, [&](sycl::handler& cgh){
            cgh.single_task([=](){
              // Do nothing
            });
          }); // Set frequency
      #endif
      // Wait for the reduction to complete
      MPI_Wait(&reduce_request, MPI_STATUS_IGNORE);
      #if HIDING == 1 
        freq_change_event.wait();
      #endif
    #else
      //TODO: Add freq change with more process
      #if HIDING == 1
        freq_change_event = mpi_queue.submit(0, VEC_ADD_FREQ, [&](sycl::handler& cgh){
              cgh.single_task([=](){
                // Do nothing
              });
            }); 
      #endif    
      // Reduce on sobel output
      MPI_Reduce(geom_output.data(), geom_reduced_data.data(),  geom_size,MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      // Waiting for the frequency change kernel 
      #if HIDING == 1
        freq_change_event.wait();
      #endif
    
    #endif


    {
        if (rank == 0)
            std::cout << "Starting vec_add ..." << std::endl;

        // TODO: Add frequency change when you run the kernel for comparing with the MPI_apporach
        buffer<int, 1> vec_add_in1_buffer(vec_add_in1.data(), range<1>(vec_add_size));
        buffer<int, 1> vec_add_in2_buffer(vec_add_in2.data(), range<1>(vec_add_size));
        buffer<int, 1> vec_add_out_buffer(vec_add_out.data(), range<1>(vec_add_size));

        for (int i = 0; i < NUM_VEC_ADD_RUNS; i++)
        {
            #if HIDING == 1
            // launch vec_add kernel with no freq. change 
                mpi_queue.submit([&](sycl::handler &cgh)
                                {
                    const sycl::accessor<int, 1, sycl::access_mode::read> in1_acc = {vec_add_in1_buffer,cgh};
                    const sycl::accessor<int, 1, sycl::access_mode::read> in2_acc = {vec_add_in2_buffer,cgh};
                    sycl::accessor<int, 1, sycl::access_mode::write> out_acc = {vec_add_out_buffer,cgh};


                    cgh.parallel_for(range<1>(vec_add_size), vec_add(vec_add_size, NUM_ITERS_VEC_ADD, in1_acc, in2_acc, out_acc)); })
                    .wait();
            #else
               mpi_queue.submit(0, VEC_ADD_FREQ, [&](sycl::handler &cgh)
                                {
                    const sycl::accessor<int, 1, sycl::access_mode::read> in1_acc = {vec_add_in1_buffer,cgh};
                    const sycl::accessor<int, 1, sycl::access_mode::read> in2_acc = {vec_add_in2_buffer,cgh};
                    sycl::accessor<int, 1, sycl::access_mode::write> out_acc = {vec_add_out_buffer,cgh};


                    cgh.parallel_for(range<1>(vec_add_size), vec_add(vec_add_size, NUM_ITERS_VEC_ADD, in1_acc, in2_acc, out_acc)); })
                    .wait();
            #endif

        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double dev_energy_per_process = mpi_queue.device_energy_consumption();

    if (rank==0){
        auto end = std::chrono::high_resolution_clock::now();   
        auto elapsed_time_app = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout<< "Total time [ms] " << elapsed_time_app.count() <<std::endl;
    }
    // auto total_time_per_process = elapsed_time_app.count();
    // int64_t total_time = 0;
    double total_energy = 0;
    // MPI_Reduce(&total_time_per_process, &total_time, sizeof(int64_t), MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dev_energy_per_process, &total_energy, sizeof(double), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // std::cout << "Total time [ms]: " << total_time << std::endl;
        std::cout << "Total energy [j]: " << total_energy << std::endl;
    }

    MPI_Finalize();

    // TODO: run the code for check
    // TODO: add frequency change with different approach (using a thread, launching a process that handle the freq.change)
    // TODO: add MPI_IReduce and MPI_IBcast

    return 0;
}