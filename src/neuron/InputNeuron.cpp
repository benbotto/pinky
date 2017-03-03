#include "InputNeuron.h"

namespace busybin {
  /**
   * Weight is ignored.  Set the single input.
   */
  void InputNeuron::pushInput(double input, double weight) {
    this->output = input;
  }

  /**
   * Get the net input for this Neuron.
   */
  double InputNeuron::getNetInput() const {
    return this->output;
  }

  /**
   * Nothing to update.
   */
  void InputNeuron::updateOutput() {
  }

  /**
   * Nothing to reset.
   */
  void InputNeuron::reset() {
  }

  /**
   * Get the name.
   */
  string InputNeuron::getName() const {
    return "InputNeuron";
  }
}

