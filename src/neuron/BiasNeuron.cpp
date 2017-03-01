#include "BiasNeuron.h"

namespace busybin {
  /**
   * Init, with a default output of 1.0.
   */
  BiasNeuron::BiasNeuron(double output) {
    this->output = output;
  }

  /**
   * Nothing to update.
   */
  void BiasNeuron::updateOutput() {
  }

  /**
   * Nothing to reset.
   */
  void BiasNeuron::reset() {
  }
}

