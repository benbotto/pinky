#ifndef _BUSYBIN_BIAS_NEURON_H_
#define _BUSYBIN_BIAS_NEURON_H_

#include "INeuron.h"

namespace busybin {
  class BiasNeuron : public INeuron {
    double output;

  public:
    BiasNeuron(double output = 1.0);
    double getOutput() const;
  };
}

#endif

