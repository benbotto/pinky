#ifndef _BUSYBIN_OUTPUT_NEURON_H_
#define _BUSYBIN_OUTPUT_NEURON_H_

#include "Neuron.h"

namespace busybin {
  class OutputNeuron : public Neuron {
  public:
    void connectTo(Neuron&, double weight);
    void feedForward() const;
  };
}

#endif

