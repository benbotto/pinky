#include "Neuron.h"

namespace busybin {
  /**
   * Add an input connection from another neuron.
   */
  void Neuron::addInput(INeuron& neuron, double weight) {
    this->inputs.push_back({&neuron, weight});
  }

  /**
   * Get the output of this Neuron.
   */
  double Neuron::getOutput() const {
    return 1.1;
  }
}

