#include "Neuron.h"

namespace busybin {
  /**
   * Add an input connection from another neuron.
   */
  Neuron& Neuron::addInput(const Neuron& neuron, double weight) {
    this->inputs.push_back({&neuron, weight});
    return *this;
  }
}

