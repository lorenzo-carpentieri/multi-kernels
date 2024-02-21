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
    kernel_freqs = read_map(is);
  }

  FreqManager() = default;

  void init(std::istream& is) {
    policy = read_policy(is);
    kernel_freqs = read_map(is);
  }

  double getAndSetFreq(const std::string& key) {
    double freq = 0;
    freq = kernel_freqs[key];
    switch (policy) {
      case FreqChangePolicy::APP:
        if (freq != 0) {
          for ( [[maybe_unused]] auto [kv, v] : kernel_freqs) {
            kernel_freqs[kv] = 0;
          }
        }
        break;
      case FreqChangePolicy::PHASE:
        kernel_freqs[key] = 0;
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

  static std::map<std::string, double> read_map(std::istream& is) {
    std::map<std::string, double> m;
    std::string key, value;
    while (is >> key >> value) {
      m[key] = atof(value.c_str());
    }
    return m;
  }

  static FreqChangePolicy read_policy(std::istream& is) {
    std::string policy;
    is >> policy;
    return policyFromString(policy);
  }

private:
  std::map<std::string, double> kernel_freqs;
  FreqChangePolicy policy;
};
