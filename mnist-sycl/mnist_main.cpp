#include <cstdio>
#include <chrono>
#include <vector>
#include <synergy.hpp>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "kernels.h"
#include "../utils/map_reader.hpp"

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

std::vector<std::string> kernel_names;
std::vector<sycl::event> event_list;
synergy::time_point_t start_time;

FreqManager freqMan {std::cin};

static inline void loaddata()
{
  mnist_load("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte",
      &train_set, &train_cnt);
  mnist_load("./t10k-images.idx3-ubyte", "./t10k-labels.idx1-ubyte",
      &test_set, &test_cnt);
}

// replace cublas function in the case n = 10
void snrm2(synergy::queue &q, const int n, float *x, float &result) {
  if (n <= 0) {
    result = 0.f;
    return;
  }
  float *r = (float*) malloc (n * sizeof(float));
  q.memcpy(r, x, n * sizeof(float)).wait();

  float sum = 0.f;
  for (int i = 0; i < n; i++) sum += r[i] * r[i];
  result = sqrtf(sum);
  free(r);
}

// Forward propagation of a single row in dataset
void forward_pass(
  synergy::queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f,
  double data[28][28])
{
  l_input.clear();
  l_c1.clear();
  l_s1.clear();
  l_f.clear();

  float input[28][28];
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      input[i][j] = data[i][j];
    }
  }
  l_input.setOutput((float *)input);

  sycl::range<1> gws (64 * 64);
  sycl::range<1> lws (64);

  // fp_preact_c1<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("fw_preact_c1"), [&] (sycl::handler &cgh) {
    auto o = (float (*)[28])l_input.output;
    auto p = (float (*)[24][24])l_c1.preact;
    auto w = (float (*)[5][5])l_c1.weight;
    cgh.parallel_for<class fw_preact_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      fp_preact_c1(item, o, p, w);
    });
  }));
  kernel_names.push_back("fw_preact_c1");

  // fp_bias_c1<<<64, 64>>>((float (*)[24][24])l_c1.preact, l_c1.bias);
  // auto c1_p_re = l_c1.preact.reinterpret<float[24][24]>(range<1>(6));
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("fw_bias_c1"), [&] (sycl::handler &cgh) {
    auto p = (float (*)[24][24])l_c1.preact;
    auto b = l_c1.bias;
    cgh.parallel_for<class fw_bias_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      fp_bias_c1(item, p, b);
    });
  }));
  kernel_names.push_back("fw_bias_c1");

  // apply_step_function<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.O);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("c1_step"), [&] (sycl::handler &cgh) {
    auto p = l_c1.preact;
    auto o = l_c1.output;
    auto l_c1_O = l_c1.O;
    cgh.parallel_for<class c1_step>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      apply_step_function(item, p, o, l_c1_O);
    });
  }));
  kernel_names.push_back("c1_step");

  // fp_preact_s1<<<64, 64>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("preact_s1"), [&] (sycl::handler &cgh) {
    auto o = (float (*)[24][24])l_c1.output;
    auto p = (float (*)[6][6])l_s1.preact;
    auto w = (float (*)[4][4])l_s1.weight;
    cgh.parallel_for<class preact_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      fp_preact_s1(item, o, p, w);
    });
  }));
  kernel_names.push_back("preact_s1");

  // fp_bias_s1<<<64, 64>>>((float (*)[6][6])l_s1.preact, l_s1.bias);
  //auto s1_p_re = l_s1.preact.reinterpret<float[6][6]>(range<1>(6));
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("fw_bias_s1"), [&] (sycl::handler &cgh) {
    auto p = (float (*)[6][6])l_s1.preact;
    auto b = l_s1.bias;
    cgh.parallel_for<class fw_bias_s1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      fp_bias_s1(item, p, b);
    });
  }));
  kernel_names.push_back("fw_bias_s1");


  // apply_step_function<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.O);
  const int l_s1_O = l_s1.O;
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("s1_step"), [&] (sycl::handler &cgh) {
    auto p = l_s1.preact;
    auto o = l_s1.output;
    cgh.parallel_for<class s1_step>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      apply_step_function(item, p, o, l_s1_O);
    });
  }));
  kernel_names.push_back("s1_step");

  //fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("preact_f"), [&] (sycl::handler &cgh) {
    auto o = (float (*)[6][6])l_s1.output;
    auto p = l_f.preact;
    auto w = (float (*)[6][6][6])l_f.weight;
    cgh.parallel_for<class preact_f>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      fp_preact_f(item, o, p, w);
    });
  }));
  kernel_names.push_back("preact_f");

  // fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("fw_bias_f"), [&] (sycl::handler &cgh) {
    auto p = l_f.preact;
    auto b = l_f.bias;
    cgh.parallel_for<class fw_bias_f>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      fp_bias_f(item, p, b);
    });
  }));
  kernel_names.push_back("fw_bias_f");

  // apply_step_function<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("f_step"), [&] (sycl::handler &cgh) {
    auto p = l_f.preact;
    auto o = l_f.output;
    auto l_f_O = l_f.O;
    cgh.parallel_for<class f_step>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      apply_step_function(item, p, o, l_f_O);
    });
  }));
  kernel_names.push_back("f_step");
}

// Back propagation to update weights
void back_pass(
  synergy::queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f)
{
  sycl::range<1> gws (64 * 64);
  sycl::range<1> lws (64);

  // bp_weight_f<<<64, 64>>>((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("bw_weight_f"), [&] (sycl::handler &cgh) {
    auto dw = (float (*)[6][6][6])l_f.d_weight;
    auto dp = l_f.d_preact;
    auto o = (float (*)[6][6])l_s1.output;
    cgh.parallel_for<class bw_weight_f>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_weight_f(item, dw, dp, o);
    });
  }));
  kernel_names.push_back("bw_weight_f");

  // bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("bw_bias_f"), [&] (sycl::handler &cgh) {
    auto b = l_f.bias;
    auto dp = l_f.d_preact;
    cgh.parallel_for<class bw_bias_f>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_bias_f(item, b, dp);
    });
  }));
  kernel_names.push_back("bw_bias_f");

  // bp_output_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("bw_output_s1"), [&] (sycl::handler &cgh) {
    auto  o = (float (*)[6][6])l_s1.d_output;
    auto  w = (float (*)[6][6][6])l_f.weight;
    auto dp = l_f.d_preact;
    cgh.parallel_for<class bw_output_s1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_output_s1(item, o, w, dp);
    });
  }));
  kernel_names.push_back("bw_output_s1");

  //bp_preact_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("bw_preact_s1"), [&] (sycl::handler &cgh) {
    auto dp = (float (*)[6][6])l_s1.d_preact;
    auto  o = (float (*)[6][6])l_s1.d_output;
    auto  p = (float (*)[6][6])l_s1.preact;
    cgh.parallel_for<class bw_preact_s1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_preact_s1(item, dp, o, p);
    });
  }));
  kernel_names.push_back("bw_preact_s1");

  // bp_weight_s1<<<64, 64>>>((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("bw_weight_s1"), [&] (sycl::handler &cgh) {
    auto dw = (float (*)[4][4])l_s1.d_weight;
    auto dp = (float (*)[6][6])l_s1.d_preact;
    auto o = (float (*)[24][24])l_c1.output;
    cgh.parallel_for<class bw_weight_s1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_weight_s1(item, dw, dp, o);
    });
  }));
  kernel_names.push_back("bw_weight_s1");

  // bp_bias_s1<<<64, 64>>>(l_s1.bias, (float (*)[6][6])l_s1.d_preact);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("bw_bias_s1"), [&] (sycl::handler &cgh) {
    auto b = l_s1.bias;
    auto dp = (float (*)[6][6])l_s1.d_preact;
    cgh.parallel_for<class bw_bias_s1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_bias_s1(item, b, dp);
    });
  }));
  kernel_names.push_back("bw_bias_s1");

  // bp_output_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("bw_output_c1"), [&] (sycl::handler &cgh) {
    auto  o = (float (*)[24][24])l_c1.d_output;
    auto  w = (float (*)[4][4])l_s1.weight;
    auto dp = (float (*)[6][6])l_s1.d_preact;
    cgh.parallel_for<class bw_output_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_output_c1(item, o, w, dp);
    });
  }));
  kernel_names.push_back("bw_output_c1");

  // bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("bw_preact_c1"), [&] (sycl::handler &cgh) {
    auto dp = (float (*)[24][24])l_c1.d_preact;
    auto  o = (float (*)[24][24])l_c1.d_output;
    auto p = (float (*)[24][24])l_c1.preact;
    cgh.parallel_for<class bw_preact_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_preact_c1(item, dp, o, p);
    });
  }));
  kernel_names.push_back("bw_preact_c1");

  // bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("bw_weight_c1"), [&] (sycl::handler &cgh) {
    auto dw = (float (*)[5][5])l_c1.d_weight;
    auto dp = (float (*)[24][24])l_c1.d_preact;
    auto o = (float (*)[28])l_input.output;
    cgh.parallel_for<class bw_weight_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_weight_c1(item, dw, dp, o);
    });
  }));
  kernel_names.push_back("bw_weight_c1");

  // bp_bias_c1<<<64, 64>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("bw_bias_c1"), [&] (sycl::handler &cgh) {
    auto b = l_c1.bias;
    auto dp = (float (*)[24][24])l_c1.d_preact;
    cgh.parallel_for<class bw_bias_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_bias_c1(item, b, dp);
    });
  }));
  kernel_names.push_back("bw_bias_c1");

  // apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("l_f_grad"), [&] (sycl::handler &cgh) {
    auto w = l_f.weight;
    auto dw = l_f.d_weight;
    auto l_f_mn = l_f.M * l_f.N;
    cgh.parallel_for<class l_f_grad>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      apply_grad(item, w, dw, l_f_mn);
    });
  }));
  kernel_names.push_back("l_f_grad");

  // apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("l_s1_grad"), [&] (sycl::handler &cgh) {
    auto w = l_s1.weight;
    auto dw = l_s1.d_weight;
    auto l_s1_mn = l_s1.M * l_s1.N;
    cgh.parallel_for<class l_s1_grad>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      apply_grad(item, w, dw, l_s1_mn);
    });
  }));
  kernel_names.push_back("l_s1_grad");

  // apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
  event_list.push_back(q.submit(0, freqMan.getAndSetFreq("l_c1_grad"), [&] (sycl::handler &cgh) {
    auto w = l_c1.weight;
    auto dw = l_c1.d_weight;
    auto l_c1_mn = l_c1.M * l_c1.N;
    cgh.parallel_for<class l_c1_grad>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      apply_grad(item, w, dw, l_c1_mn);
    });
  }));
  kernel_names.push_back("l_c1_grad");
}

static void learn(
  synergy::queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f,
  int iter)
{
  float err;
  // fprintf(stdout ,"Learning\n");
  std::cerr << "Learning" << std::endl;

  while (iter < 0 || iter-- > 0) {
    err = 0.0f;

    for (unsigned int i = 0; i < train_cnt; i += 1000) { // changed from 1 to 50 to reduce the number of iterations
      float tmp_err;  

      q.submit(0, freqMan.getAndSetFreq("phase1"), [&](sycl::handler& cgh) {

      }).wait();

      double start_forward_pass_energy = q.get_synergy_device().get_energy_usage();
      forward_pass(q, l_input, l_c1, l_s1, l_f, train_set[i].data);
     
      double end_forward_pass_energy = q.get_synergy_device().get_energy_usage();
      std::cerr<<"forward_pass_learn_energy: " << (end_forward_pass_energy -start_forward_pass_energy) / 1e6 << std::endl;
      l_f.bp_clear();
      l_s1.bp_clear();
      l_c1.bp_clear();

      // Euclid distance of train_set[i]
      sycl::range<1> gws (10);
      sycl::range<1> lws (1);
      // makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
      event_list.push_back(q.submit(0, freqMan.getAndSetFreq("err"), [&] (sycl::handler &cgh) {
        auto dp = l_f.d_preact;
        auto o = l_f.output;
        auto train_set_label = train_set[i].label;
        cgh.parallel_for<class err>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          makeError(item, dp, o, train_set_label, 10);
        });
      }));
      kernel_names.push_back("err");
      double start_back_pass_energy = q.get_synergy_device().get_energy_usage();

      snrm2(q, 10, l_f.d_preact, tmp_err);
      err += tmp_err;


      q.submit(0, freqMan.getAndSetFreq("phase2"), [&](sycl::handler& cgh) {

      }).wait();

      back_pass(q, l_input, l_c1, l_s1, l_f);
      double end_back_pass_energy = q.get_synergy_device().get_energy_usage();
      std::cerr<<"back_pass_learn_energy: " << (end_back_pass_energy -start_back_pass_energy) / 1e6 << std::endl;
    }

    err /= train_cnt;
    // fprintf(stdout, "error: %e\n", err);
    std::cerr << "error: " << err << std::endl;

    if (err < threshold) {
      // fprintf(stdout, "Training complete, error less than threshold\n\n");
      std::cerr << "Training complete, error less than threshold" << std::endl;
      break;
    }
  }
}


// Returns label of given data (0-9)
static unsigned int classify(
  synergy::queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f,
  double data[28][28])
{
  float res[10];
  double start_forward_pass_energy = q.get_synergy_device().get_energy_usage();

  forward_pass(q, l_input, l_c1, l_s1, l_f, data);
  double end_forward_pass_energy = q.get_synergy_device().get_energy_usage();
  std::cerr<<"forward_pass_classify_energy: " << (end_forward_pass_energy -start_forward_pass_energy) / 1e6 << std::endl;
  unsigned int max = 0;

  q.memcpy(res, l_f.output, sizeof(float) * 10).wait();

  for (int i = 1; i < 10; ++i) {
    if (res[max] < res[i]) {
      max = i;
    }
  }

  return max;
}

// Perform forward propagation of test data
static void test(
  synergy::queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f)
{
  // fprintf(stdout ,"Testing\n");
  std::cerr << "Testing" << std::endl;

  int error = 0;

  q.submit(0, freqMan.getAndSetFreq("phase3"), [&](sycl::handler& cgh) {

  }).wait();

  for (unsigned int i = 0; i < test_cnt; i += 1000) { // changed from 1 to 1000 to reduce the number of iterations
    if (classify(q, l_input, l_c1, l_s1, l_f, test_set[i].data)
        != test_set[i].label) {
      ++error;
    }
  }

  // fprintf(stdout, "Error Rate: %.2lf%%\n",
  //     double(error) / double(test_cnt) * 100.0);
  std::cerr << "Error Rate: " << double(error) / double(test_cnt) * 100.0 << "%" << std::endl;
}

int main(int argc, const  char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }

  const int iter = atoi(argv[1]);
  srand(123);
  loaddata();

  synergy::queue q(sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order()});
// #ifdef USE_GPU
//   sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
// #else
//   sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
// #endif

  Layer l_input (q, 0, 0, 28*28);
  Layer l_c1 (q, 5*5, 6, 24*24*6);
  Layer l_s1 (q, 4*4, 1, 6*6*6);
  Layer l_f (q, 6*6*6, 10, 10);

  auto t1 = std::chrono::high_resolution_clock::now();
  start_time = std::chrono::high_resolution_clock::now();
  learn(q, l_input, l_c1, l_s1, l_f, iter);
  test(q, l_input, l_c1, l_s1, l_f);
  auto t2 = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  // printf("Total time (learn + test) %lf secs \n", total_time / 1.0e6);
  std::cerr << "Total time (learn + test) " << total_time / 1.0e6 << " secs" << std::endl;
  std::cerr << "Total events: " << event_list.size() << std::endl;

#ifdef SYNERGY_KERNEL_PROFILING
  auto host_energy = q.host_energy_consumption();
  synergy::Profiler<double> synergy_profiler {q, event_list, start_time};
  std::cout << "kernel_name,host_energy[j],memory_freq [MHz],core_freq [MHz],times[ms],kernel_energy[j],total_real_time[ms],sum_kernel_times[ms],total_device_energy[j],sum_kernel_energy[j]" << std::endl;
  for (int i = 0; i < kernel_names.size(); i++) {
    std::cout << kernel_names[i] << "," << host_energy << ",";
    synergy_profiler.print_all_profiling_info(i);
  }
#else
  auto host_energy = q.host_energy_consumption();
  auto device_energy = q.device_energy_consumption();
  std::cout << "Total time [ms]: " << total_time << std::endl;
  std::cout << "Host energy [J]: " << host_energy << std::endl;
  std::cout << "Device energy [J]: " << device_energy << std::endl;
#endif

  return 0;
}
