
#include <../../utils/map_reader.hpp>
#include <synergy.hpp>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>
#include <memory>
#include "bitmap.h"


inline void swap(sycl::float4 A[], int i, int j) {
  A[i] = fmin(A[i], A[j]);
  A[j] = fmax(A[i], A[j]);
}

class Median {
public:
  synergy::queue& q;
  size_t size;
  std::vector<sycl::float4> input;
  std::vector<sycl::float4> output;
  std::shared_ptr<sycl::buffer<sycl::float4, 2>> input_buf;
  std::shared_ptr<sycl::buffer<sycl::float4, 2>> output_buf;

  Median(synergy::queue& q, size_t size) : q{q}, size{size} {
    input.resize(size * size);
    // TODO: specify image path
    load_bitmap_mirrored("/home/lcarpent/energy-workspace/multi-kernels/SYnergy/samples/freq_overhead/Brommy.bmp", size, input);
    output.resize(size * size);
    input_buf = std::make_shared<sycl::buffer<sycl::float4, 2>>(input.data(), sycl::range<2>{size, size});
    output_buf = std::make_shared<sycl::buffer<sycl::float4, 2>>(output.data(), sycl::range<2>{size, size});
  }

  sycl::event operator() () {
    return q.submit([&](sycl::handler& cgh) {
      auto in = input_buf->get_access<sycl::access::mode::read>(cgh);
      auto out = output_buf->get_access<sycl::access::mode::discard_write>(cgh);
      sycl::range<2> ndrange{size, size};

      cgh.parallel_for<class MedianFilterBenchKernel>(
      ndrange, [in, out, size_ = size](sycl::id<2> gid) {
        int x = gid[0];
        int y = gid[1];

        sycl::float4 window[9];

        int k = 0;
        for(int i = -1; i < 2; i++)
          for(int j = -1; j < 2; j++) {
            uint xs = sycl::min(
                sycl::max(x + j, 0), static_cast<int>(size_ - 1)); // borders are handled here with extended values
            uint ys = sycl::min(sycl::max(y + i, 0), static_cast<int>(size_ - 1));
            window[k] = in[{xs, ys}];
            k++;
          }

        // (channel-wise) median selection using bitonic sorting
        // the following network is used (Bose-Nelson algorithm):
        // [[0,1],[2,3],[4,5],[7,8]]
        // [[0,2],[1,3],[6,8]]
        // [[1,2],[6,7],[5,8]]
        // [[4,7],[3,8]]
        // [[4,6],[5,7]]
        // [[5,6],[2,7]]
        // [[0,5],[1,6],[3,7]]
        // [[0,4],[1,5],[3,6]]
        // [[1,4],[2,5]]
        // [[2,4],[3,5]]
        // [[3,4]]
        // se also http://pages.ripco.net/~jgamble/nw.html
        swap(window, 0, 1);
        swap(window, 2, 3);
        swap(window, 0, 2);
        swap(window, 1, 3);
        swap(window, 1, 2);
        swap(window, 4, 5);
        swap(window, 7, 8);
        swap(window, 6, 8);
        swap(window, 6, 7);
        swap(window, 4, 7);
        swap(window, 4, 6);
        swap(window, 5, 8);
        swap(window, 5, 7);
        swap(window, 5, 6);
        swap(window, 0, 5);
        swap(window, 0, 4);
        swap(window, 1, 6);
        swap(window, 1, 5);
        swap(window, 1, 4);
        swap(window, 2, 7);
        swap(window, 3, 8);
        swap(window, 3, 7);
        swap(window, 2, 5);
        swap(window, 2, 4);
        swap(window, 3, 6);
        swap(window, 3, 5);
        swap(window, 3, 4);

        out[gid] = window[4];
      });
    });
  }
};



int main(int argc, char * argv []){
  
  if (argc != 2){
    std::cerr << "Usage ./median <size>" << std::endl;
    return 0; 
  }
  const size_t size = atoi(argv[1]); // size of the matrix will  be size x size
  
  synergy::queue warm_up_q{sycl::gpu_selector_v};
  /******* START Frequency Change ********/  
  // Create frequency manager for handling frequency change
  FreqManager freqMan = FreqManager(std::cin); // Read the frequency configuration file from the standard input

  warm_up_q.submit(0, freqMan.getAndSetFreq("median"), [&](sycl::handler& cgh) {
      
  }).wait();
  /******* END Frequency Change ********/  

  Median warm_up_median_kernel{warm_up_q, size};
  
  /******* START Warm up ********/
  sycl::event e = warm_up_median_kernel();
  e.wait();
  auto warm_up_host_energy = warm_up_q.host_energy_consumption(); // Host energy consumed for warming up
  //***** END Warm up ***** */


  std::vector<sycl::event> events; // Array of events with all the kernels that should be profiled
  std::vector<synergy::time_point_t> start_times;
  std::vector<std::string> kernel_names; 

 
  /******* START Kernel Execution for 5 seconds ********/  

  synergy::queue q {sycl::gpu_selector_v};
  // Change the frequency
  q.submit(0, freqMan.getAndSetFreq("median"), [&](sycl::handler& cgh) {
      
  }).wait();

  Median median_kernel{q, size};
  double total_chain_time_ms = 0.0;
  int chain_size = 0;
  auto start = synergy::wall_clock_t::now();
  // Run Median kernel consecutevly for 5 seconds
  while (total_chain_time_ms < 5000.0) { // Run for approximately 5 seconds
    kernel_names.push_back("median"); // profile all the kernels
    start_times.push_back(synergy::wall_clock_t::now());
    sycl::event chain_e = median_kernel();
    chain_e.wait();
    events.push_back(chain_e);
    auto chain_start = chain_e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto chain_end = chain_e.get_profiling_info<sycl::info::event_profiling::command_end>();
    total_chain_time_ms += (chain_end - chain_start) / 1000000.0;
    chain_size++;
  }

  /******* END Kernel Execution for 5 seconds ********/  

  auto host_energy = (q.host_energy_consumption() - warm_up_host_energy);
  

  #ifdef SYNERGY_KERNEL_PROFILING
    synergy::Profiler<double> synergy_profiler(q, events, start);
    std::cout << "kernel_name,host_energy[j],chain_size,memory_freq [MHz],core_freq [MHz],times[ms],kernel_energy[j],total_real_time[ms],sum_kernel_times[ms],total_device_energy[j],sum_kernel_energy[j]" << std::endl;
    for(int i = 0; i < events.size(); i++){
        std::string s = kernel_names[i];
        std::cout << s << "," << host_energy << "," << chain_size << ",";
        synergy_profiler.print_all_profiling_info(i);
    }
  #endif

  return 0;    

}