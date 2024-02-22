#include <map> 
#include <string>
#include <iostream>


class FreqManager {
public:
  enum class FreqChangePolicy {
    APP,
    PHASE,
    KERNEL
  };

  FreqManager(std::istream& is) {
    try {
      init(is);
    } catch (std::exception e) {
      std::cerr << e.what() << std::endl;
    }
  }

  FreqManager() = default;

  void init(std::istream& is) {
    read_policy(is);
    read_map(is);
  }

  double getAndSetFreq(const std::string& key) {
    double freq = 0;
    if (kernel_freqs.find(key) == kernel_freqs.end()) {
      std::cerr << "Kernel name '" + key + "' not found" << std::endl;
      return 0;
    }
    freq = kernel_freqs[key];
    switch (policy) {
      case FreqChangePolicy::APP:
        if (freq != 0) {
          for (auto [kv, _] : kernel_freqs) {
            kernel_freqs[kv] = 0;
          }
        }
        break;
      case FreqChangePolicy::PHASE:
        if (!keep_freq[key]) {
          kernel_freqs[key] = 0;
        }
        break;
      case FreqChangePolicy::KERNEL:
        break;
    }
    return freq;
  }

  static FreqChangePolicy policyFromString(const std::string& s) {
    if (s == "APP") {
      return FreqChangePolicy::APP;
    } else if (s == "PHASE") {
      return FreqChangePolicy::PHASE;
    } else if (s == "KERNEL") {
      return FreqChangePolicy::KERNEL;
    } else {
      throw std::runtime_error("Unknown FreqChangePolicy");
    }
  }

  static FreqChangePolicy policyFromString(const char* s) {
    return policyFromString(std::string(s));
  }

private:
  void read_map(std::istream& is) {
    std::string key, value, keep;
    while (is >> key >> value >> keep) {
      keep_freq[key] = keep == "KEEP";
      kernel_freqs[key] = atof(value.c_str());
    }
  }

  void read_policy(std::istream& is) { 
    std::string policy;
    is >> policy;
    this->policy = policyFromString(policy);
  }

  std::map<std::string, double> kernel_freqs;
  std::map<std::string, bool> keep_freq;
  FreqChangePolicy policy;
};
