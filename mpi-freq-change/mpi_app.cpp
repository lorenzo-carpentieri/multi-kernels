#include <iostream>
#include <sycl/sycl.hpp>
#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <synergy.hpp>

using namespace sycl;

#define NUM_MERSE_RUN 1
#define NUM_SOBEL_RUN 1
#define NUM_ITERS_SOBEL 2000
#define NUM_ITERS_MERSE 50000

//Merse_twister parameters
#define MT_RNG_COUNT 4096
#define MT_MM 9
#define MT_NN 19
#define MT_WMASK 0xFFFFFFFFU
#define MT_UMASK 0xFFFFFFFEU
#define MT_LMASK 0x1U
#define MT_SHIFT0 12
#define MT_SHIFTB 7
#define MT_SHIFTC 15
#define MT_SHIFT1 18
#define PI 3.14159265358979f

// Size of the vector
constexpr size_t N = 2048;

// Define a custom datatype for float4
void create_mpi_float4_type(MPI_Datatype* mpi_type) {
    MPI_Type_contiguous(4, MPI_FLOAT, mpi_type);
    MPI_Type_commit(mpi_type);
}


// generate random floating point numbers in the range [0,1]
float randomFloat() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}
// init merse vector
void init_merse(std::vector<uint> &ma,
                std::vector<uint> &c,
                std::vector<uint> &b,
                std::vector<uint> &seed){
    
    for(uint i = 0; i < ma.size(); ++i) {
      ma[i] = i;
      b[i] = i;
      c[i] = i;
      seed[i] = i;
    }

}


// init sobel vector
void init_sobel(std::vector<sycl::float4> &input,std::vector<sycl::float4> &output ){
    for (int i = 0; i < N; ++i) {
        input[i] = randomFloat();
        output[i] = 0;
    }
}

// Sobel 3
class sobel {
private:
  int size;
  int num_iters;
  const sycl::accessor<sycl::float4, 2, sycl::access_mode::read> in;
  sycl::accessor<sycl::float4, 2, sycl::access_mode::discard_write> out;

public:
  sobel(int size, int num_iters, const sycl::accessor<sycl::float4, 2, sycl::access_mode::read> in,
      sycl::accessor<sycl::float4, 2, sycl::access_mode::discard_write> out)
      : size(size), num_iters(num_iters), in(in), out(out) {}

  void operator()(sycl::id<2> gid) const {
    const float kernel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int x = gid[0];
    int y = gid[1];
    // num itesrs for sobel is 2000
    for(size_t i = 0; i < num_iters; i++) {
      sycl::float4 Gx = sycl::float4(0, 0, 0, 0);
      sycl::float4 Gy = sycl::float4(0, 0, 0, 0);
      const int radius = 3;

      // constant-size loops in [0,1,2]
      for(int x_shift = 0; x_shift < 3; x_shift++) {
        for(int y_shift = 0; y_shift < 3; y_shift++) {
          // sample position
          uint xs = x + x_shift - 1; // [x-1,x,x+1]
          uint ys = y + y_shift - 1; // [y-1,y,y+1]
          // for the same pixel, convolution is always 0
          if(x == xs && y == ys)
            continue;
          // boundary check
          if(xs < 0 || xs >= size || ys < 0 || ys >= size)
            continue;

          // sample color
          sycl::float4 sample = in[{xs, ys}];

          // convolution calculation
          int offset_x = x_shift + y_shift * radius;
          int offset_y = y_shift + x_shift * radius;

          float conv_x = kernel[offset_x];
          sycl::float4 conv4_x = sycl::float4(conv_x);
          Gx += conv4_x * sample;

          float conv_y = kernel[offset_y];
          sycl::float4 conv4_y = sycl::float4(conv_y);
          Gy += conv4_y * sample;
        }
      }
      // taking root of sums of squares of Gx and Gy
      sycl::float4 color = hypot(Gx, Gy);
      sycl::float4 minval = sycl::float4(0.0, 0.0, 0.0, 0.0);
      sycl::float4 maxval = sycl::float4(1.0, 1.0, 1.0, 1.0);
      out[gid] = clamp(color, minval, maxval);
    }
  }
};


// mersetwister
class merse_twister {
private:
  int size;
  int num_iters;
  const sycl::accessor<uint, 1, sycl::access_mode::read> ma_acc;
  const sycl::accessor<uint, 1, sycl::access_mode::read> b_acc;
  const sycl::accessor<uint, 1, sycl::access_mode::read> seed_acc;
  const sycl::accessor<uint, 1, sycl::access_mode::read> c_acc;
  sycl::accessor<sycl::float4, 1, sycl::access_mode::read_write> result_acc;



public:
  merse_twister(int size, int num_iters, 
        const sycl::accessor<uint, 1, sycl::access_mode::read> ma_acc,
        const sycl::accessor<uint, 1, sycl::access_mode::read> b_acc,
        const sycl::accessor<uint, 1, sycl::access_mode::read> c_acc,
        const sycl::accessor<uint, 1, sycl::access_mode::read> seed_acc,
        sycl::accessor<sycl::float4, 1, sycl::access_mode::read_write> result_acc)
      : size(size), num_iters(num_iters), ma_acc(ma_acc), b_acc(b_acc), c_acc(c_acc), seed_acc(seed_acc), result_acc(result_acc) {}

  void operator()(sycl::id<1> id) const {
        int gid = id[0];

        if(gid >= size)
          return;
        for(size_t i = 0; i < num_iters; i++) {
          int iState, iState1, iStateM;
          unsigned int mti, mti1, mtiM, x;
          unsigned int matrix_a, mask_b, mask_c;

          unsigned int mt[MT_NN]; // FIXME

          matrix_a = ma_acc[gid];
          mask_b = b_acc[gid];
          mask_c = c_acc[gid];

          mt[0] = seed_acc[gid];
          for(iState = 1; iState < MT_NN; iState++)
            mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

          iState = 0;
          mti1 = mt[0];

          float tmp[5];
          for(int i = 0; i < 4; ++i) {
            iState1 = iState + 1;
            iStateM = iState + MT_MM;
            if(iState1 >= MT_NN)
              iState1 -= MT_NN;
            if(iStateM >= MT_NN)
              iStateM -= MT_NN;
            mti = mti1;
            mti1 = mt[iState1];
            mtiM = mt[iStateM];

            x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
            x = mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);

            mt[iState] = x;
            iState = iState1;

            // Tempering transformation
            x ^= (x >> MT_SHIFT0);
            x ^= (x << MT_SHIFTB) & mask_b;
            x ^= (x << MT_SHIFTC) & mask_c;
            x ^= (x >> MT_SHIFT1);

            tmp[i] = ((float)x + 1.0f) / 4294967296.0f;
          }

          sycl::float4 val;
          val.s0() = tmp[0];
          val.s1() = tmp[1];
          val.s2() = tmp[2];
          val.s3() = tmp[3];

          result_acc[gid] = val;
        }
  }
};

// Print the vector
void print_vector(float* data, size_t size, size_t rank) {
    std::cout << "Process " << rank << ": [" << data[N-1] <<"]" << std::endl;

}

int main(int argc, char* argv[]) {
    int num_processes, rank;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const synergy::frequency SOBEL_FREQ = 350;
    const synergy::frequency MERSE_FREQ = 1200;

    // Initialize SYCL queue for each process
    const auto &devices = sycl::device::get_devices();
    if (devices.empty()) {
      std::cerr << "No SYCL devices available" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    std::vector<device> gpu_devices;

    for(size_t i = 0; i < devices.size(); i++){
      if (devices[i].has(sycl::aspect::gpu) && devices[i].get_platform().get_info<sycl::info::platform::name>().find("OpenCL") == std::string::npos)
        gpu_devices.push_back(devices[i]);
    }
    // create vector for sobel and merse for each process 
    // merse_twister vector
    const size_t merse_size = 524288;
    std::vector<uint> merse_ma(merse_size);
    std::vector<uint> merse_b(merse_size);
    std::vector<uint> merse_c(merse_size);
    std::vector<uint> merse_seed(merse_size);
    std::vector<sycl::float4> merse_result(merse_size);

    // sobel
    const size_t sobel_size = 3072;
    std::vector<sycl::float4> sobel_input(sobel_size*sobel_size);
    std::vector<sycl::float4> sobel_output(sobel_size*sobel_size);

    // init_soble only rank_0
    // init_merse only rank_0
    // Broadcast to all process

    
    // The rank 0 process send the data to all other process
    if(rank==0){
        init_merse(merse_ma,merse_b, merse_c, merse_seed);
        init_sobel(sobel_input, sobel_output);
        // std::fill(input_data.begin(), input_data.end(), 1);
        // std::fill(result_data.begin(), result_data.end(), 0);
    }

    // Create SYnergy queue
    auto mpi_queue = synergy::queue(gpu_devices[rank]);
    // sycl event that will be associated to the frequency change kernel
    #if HIDING == 1
      sycl::event freq_change_event;
    #endif

    // MPI_IBcast implementation
    #ifdef WITH_MPI_ASYNCH
      // Broadcasts with MPI_Ibcast
      MPI_Request request[5];
      MPI_Ibcast(merse_ma.data(), merse_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD, &request[0]);
      MPI_Ibcast(merse_b.data(), merse_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD, &request[1]);
      MPI_Ibcast(merse_c.data(), merse_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD, &request[2]);
      MPI_Ibcast(merse_seed.data(), merse_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD, &request[3]);
      MPI_Ibcast(sobel_input.data(), sobel_size * sobel_size * sizeof(sycl::float4), MPI_BYTE, 0, MPI_COMM_WORLD, &request[4]);
      // Hiding frequency change in MPI_Ibcast
      #if HIDING == 1 
            freq_change_event = mpi_queue.submit(0, SOBEL_FREQ, [&](sycl::handler& cgh){
              cgh.single_task([=](){
                // Do nothing
              });
            }); // Set frequency
      #endif
      MPI_Waitall(5, request, MPI_STATUSES_IGNORE);
      #if HIDING == 1
        freq_change_event.wait();
      #endif
    // MPI_Bcast implementation
    #else
      #if HIDING == 1
        freq_change_event = mpi_queue.submit(0, SOBEL_FREQ, [&](sycl::handler& cgh){
              cgh.single_task([=](){
                // Do nothing
              });
            }); 
      #endif     
      //TODO: Having more process for frequency change 
      // Wait for all broadcasts to complete
      MPI_Bcast(merse_ma.data(), merse_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      MPI_Bcast(merse_b.data(), merse_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      MPI_Bcast(merse_c.data(), merse_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      MPI_Bcast(merse_seed.data(), merse_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      // Bcast for sobel
      MPI_Bcast(sobel_input.data(), sobel_size * sobel_size * sizeof(sycl::float4), MPI_BYTE, 0, MPI_COMM_WORLD);
      // Waiting the sycl kernel for the frequency change
      #if HIDING == 1
        freq_change_event.wait();
      #endif
    #endif
    
    {   
        #if HIDING == 0 
          mpi_queue.submit(0, SOBEL_FREQ, [&](sycl::handler& cgh){
            cgh.single_task([=](){
              // Do nothing
            });
          }).wait(); // Set frequency
        #endif
        buffer<sycl::float4, 2> sobel_input_buffer(sobel_input.data(), range<2>(sobel_size , sobel_size));
        buffer<sycl::float4, 2> sobel_output_buffer(sobel_output.data(), range<2>(sobel_size , sobel_size));

        for(int i = 0; i < NUM_SOBEL_RUN; i++){
            //launch sobel kernel
            mpi_queue.submit([&](sycl::handler& cgh) {
                const sycl::accessor<sycl::float4, 2, sycl::access_mode::read> input_accessor = {sobel_input_buffer,cgh};
                 sycl::accessor<sycl::float4, 2, sycl::access_mode::discard_write> output_accessor = {sobel_output_buffer,cgh};
                cgh.parallel_for(range<2>(sobel_size, sobel_size), sobel(sobel_size, NUM_ITERS_SOBEL, input_accessor, output_accessor));
            }).wait();        
        }
    }

    std::vector<float> sobel_reduced_data(sobel_size*sobel_size*4);
    std::vector<float> sobel_output_float(sobel_size * sobel_size * 4);
    for (size_t i = 0; i < sobel_size*sobel_size; ++i) {
        sobel_output_float[i * 4] = sobel_output[i].x();
        sobel_output_float[i * 4 + 1] = sobel_output[i].y();
        sobel_output_float[i * 4 + 2] = sobel_output[i].z();
        sobel_output_float[i * 4 + 3] = sobel_output[i].w();
    }
    #ifdef WITH_MPI_ASYNCH
      MPI_Request reduce_request;
      MPI_Ireduce(sobel_output_float.data(), sobel_reduced_data.data(), sobel_size * sobel_size * 4,
                  MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD, &reduce_request);
      #if HIDING == 1 
          mpi_queue.submit(0, MERSE_FREQ, [&](sycl::handler& cgh){
            cgh.single_task([=](){
              // Do nothing
            });
          }).wait(); // Set frequency
      #endif
      // Wait for the reduction to complete
      MPI_Wait(&reduce_request, MPI_STATUS_IGNORE);
    #else
      //TODO: Add freq change with more process
      #if HIDING == 1
        freq_change_event = mpi_queue.submit(0, MERSE_FREQ, [&](sycl::handler& cgh){
              cgh.single_task([=](){
                // Do nothing
              });
            }); 
      #endif    
      // Reduce on sobel output
      MPI_Reduce(sobel_output_float.data(), sobel_reduced_data.data(),  sobel_size * sobel_size * 4,MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      // Waiting for the frequency change kernel 
      #if HIDING == 1
        freq_change_event.wait();
      #endif
    
    #endif

    {
        #if HIDING == 0 
          mpi_queue.submit(0, MERSE_FREQ, [&](sycl::handler& cgh){
            cgh.single_task([=](){
              // Do nothing
            });
          }).wait(); // Set frequency
        #endif
        //TODO: Add frequency change when you run the kernel for comparing with the MPI_apporach
        buffer<uint, 1> merse_ma_buffer(merse_ma.data(), range<1>(merse_size));
        buffer<uint, 1> merse_b_buffer(merse_b.data(), range<1>(merse_size));
        buffer<uint, 1> merse_c_buffer(merse_c.data(), range<1>(merse_size));
        buffer<uint, 1> merse_seed_buffer(merse_seed.data(), range<1>(merse_size));
        buffer<sycl::float4, 1> merse_result_buffer(merse_result.data(), range<1>(merse_size));
    
        for(int i = 0; i < NUM_MERSE_RUN; i++){
            //launch sobel kernel
            mpi_queue.submit([&](sycl::handler& cgh) {
                sycl::accessor<sycl::float4, 1, sycl::access_mode::read_write> result_accessor = {merse_result_buffer,cgh};
                const sycl::accessor<uint, 1, sycl::access_mode::read> ma_accessor = {merse_ma_buffer,cgh};
                const sycl::accessor<uint, 1, sycl::access_mode::read> b_accessor = {merse_b_buffer,cgh};
                const sycl::accessor<uint, 1, sycl::access_mode::read> c_accessor = {merse_c_buffer,cgh};
                const sycl::accessor<uint, 1, sycl::access_mode::read> seed_accessor = {merse_seed_buffer,cgh};


                cgh.parallel_for(range<1>(merse_size), merse_twister(merse_size, NUM_ITERS_MERSE, ma_accessor, b_accessor, c_accessor, seed_accessor, result_accessor));
            }).wait();        
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    //TODO: run the code for check
    //TODO: add frequency change with different approach (using a thread, launching a process that handle the freq.change)
    //TODO: add MPI_IReduce and MPI_IBcast

    return 0;
}