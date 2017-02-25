#include "BiasNeuron.h"

namespace busybin {
  /**
   * Init, with a default output of 1.0.
   */
  BiasNeuron::BiasNeuron(double output) : output(output) {
  }

  /**
   * The output is static.
   */
  double BiasNeuron::getOutput() const {
    return this->output;
  }
}

