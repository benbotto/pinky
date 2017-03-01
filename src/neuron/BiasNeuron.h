#ifndef _BUSYBIN_BIAS_NEURON_H_
#define _BUSYBIN_BIAS_NEURON_H_

#include "Neuron.h"

namespace busybin {
  class BiasNeuron : public Neuron {
  public:
    BiasNeuron(double output = 1.0);
    void updateOutput();
    void reset();
  };
}

#endif

