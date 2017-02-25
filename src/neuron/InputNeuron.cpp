#include "InputNeuron.h"

namespace busybin {
  /**
   * Input defaults to 0.
   */
  InputNeuron::InputNeuron() : input(0) {
  }

  /**
   * Set the input.
   */
  void InputNeuron::setInput(double input) {
    this->input = input;
  }

  /**
   * The output is the same as the input on first-layer neurons.
   */
  double InputNeuron::getOutput() const {
    return this->input;
  }
}

