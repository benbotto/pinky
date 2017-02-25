#ifndef _BUSYBIN_INPUT_NEURON_H_
#define _BUSYBIN_INPUT_NEURON_H_

#include "Neuron.h"

namespace busybin {
  class InputNeuron : public INeuron {
    double input;

  public:
    InputNeuron();
    void setInput(double input);
    double getOutput() const;
  };
}

#endif

