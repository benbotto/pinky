#ifndef _BUSYBIN_INPUT_NEURON_H_
#define _BUSYBIN_INPUT_NEURON_H_

#include "Neuron.h"

namespace busybin {
  class InputNeuron : public Neuron {
  public:
    void pushInput(double input, double weight = 1);
    double getNetInput() const;
    void updateOutput();
    void reset();
    string getName() const;
  };
}

#endif

