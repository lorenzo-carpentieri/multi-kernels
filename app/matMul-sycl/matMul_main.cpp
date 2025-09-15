#include <../../utils/map_reader.hpp>
#include <synergy.hpp>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>
#include <memory>

class MatMul {
public:
  synergy::queue& q;
  size_t size;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
  std::shared_ptr<sycl::buffer<int, 2>> a_buf;
  std::shared_ptr<sycl::buffer<int, 2>> b_buf; 
  std::shared_ptr<sycl::buffer<int, 2>> c_buf;

  MatMul(synergy::queue& q, size_t size) : q{q}, size{size} {
    a.resize(size * size);
    b.resize(size * size);
    c.resize(size * size);
    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 1);
    std::fill(c.begin(), c.end(), 0);
    a_buf = std::make_shared<sycl::buffer<int, 2>>(a.data(), sycl::range<2>{size, size});
    b_buf = std::make_shared<sycl::buffer<int, 2>>(b.data(), sycl::range<2>{size, size});
    c_buf = std::make_shared<sycl::buffer<int, 2>>(c.data(), sycl::range<2>{size, size});
  }

  sycl::event operator()() {
    return q.submit([&](sycl::handler& h) {
      sycl::accessor a_acc{*(a_buf.get()), h, sycl::read_only};
      sycl::accessor b_acc{*(b_buf.get()), h, sycl::read_only};
      sycl::accessor c_acc{*(c_buf.get()), h, sycl::read_write};

      sycl::range<2> grid{size, size};
      sycl::range<2> block{size < 32 ? size : 32, size < 32 ? size : 32};

      h.parallel_for(sycl::nd_range<2>(grid, block), [=, size=size](sycl::nd_item<2> idx) {
        int i = idx.get_global_id(0);
        int j = idx.get_global_id(1);
        c_acc[i][j] = 0.0f;
        for (size_t k = 0; k < size; k++) {
            c_acc[i][j] += a_acc[i][k] * b_acc[k][j];
        }
      });
    });
  }
};


int main(int argc, char * argv []){
  
  if (argc != 2){
    std::cerr << "Usage ./matMul <size>" << std::endl;
    return 0; 
  }
  const size_t size = atoi(argv[1]); // size of the matrix will  be size x size
  
  synergy::queue warm_up_q{sycl::gpu_selector_v};
  /******* START Frequency Change ********/  
  // Create frequency manager for handling frequency change
  FreqManager freqMan = FreqManager(std::cin); // Read the frequency configuration file from the standard input

  warm_up_q.submit(0, freqMan.getAndSetFreq("matMul"), [&](sycl::handler& cgh) {
      
  }).wait();
  /******* END Frequency Change ********/  

  MatMul warm_up_matmul_kernel{warm_up_q, size};
  
  /******* START Warm up ********/
  sycl::event e = warm_up_matmul_kernel();
  e.wait();
  auto warm_up_host_energy = warm_up_q.host_energy_consumption(); // Host energy consumed for warming up
  //***** END Warm up ***** */


  std::vector<sycl::event> events; // Array of events with all the kernels that should be profiled
  std::vector<synergy::time_point_t> start_times;
  std::vector<std::string> kernel_names; 

 
  /******* START Kernel Execution for 5 seconds ********/  

  synergy::queue q {sycl::gpu_selector_v};
  // Change the frequency
  q.submit(0, freqMan.getAndSetFreq("matMul"), [&](sycl::handler& cgh) {
      
  }).wait();

  MatMul matmul_kernel{q, size};
  double total_chain_time_ms = 0.0;
  int chain_size = 0;
  auto start = synergy::wall_clock_t::now();
  // Run matMul kernel consecutevly for 5 seconds
  while (total_chain_time_ms < 5000.0) { // Run for approximately 5 seconds
    kernel_names.push_back("matMul"); // profile all the kernels
    start_times.push_back(synergy::wall_clock_t::now());
    sycl::event chain_e = matmul_kernel();
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