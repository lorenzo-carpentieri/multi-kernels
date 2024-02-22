#include <map> 
#include <string>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>


class FreqManager {
public:
  enum class FreqChangePolicy {
    APP,
    PHASE,
    KERNEL
  };

  FreqManager(std::istream& is) { 
    init(is);
  }

  FreqManager() = default;

  void init(std::istream& is) {
    if (isInputAvailable()) {
      read_policy(is);
      read_map(is);
      enabled = true;
    } else {
      std::cerr << "Frequency scaling disabled" << std::endl;
      enabled = false;
    }
  }

  double getAndSetFreq(const std::string& key) {
    if (!enabled) return 0;

    double freq = 0;
    freq = kernel_freqs[key];
    std::cerr << "Retrieved freq " << freq << " on key " << key << std::endl;
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
      kernel_freqs[key] = std::stod(value);
    }
  }

  void read_policy(std::istream& is) { 
    std::string policy;
    is >> policy;
    this->policy = policyFromString(policy);
  }

  // Function to check if input is available without blocking
  bool isInputAvailable() {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    
    int ret = select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
    if (ret == -1) {
      throw std::runtime_error("Error in select()");
    }
    return ret > 0;
  }

  std::map<std::string, double> kernel_freqs;
  std::map<std::string, bool> keep_freq;
  bool enabled;
  FreqChangePolicy policy;
};
