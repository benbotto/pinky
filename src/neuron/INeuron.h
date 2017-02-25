#ifndef _BUSYBIN_INEURON_H_
#define _BUSYBIN_INEURON_H_

namespace busybin {
  class INeuron {
  public:
    virtual double getOutput() const = 0;
  };
}

#endif

