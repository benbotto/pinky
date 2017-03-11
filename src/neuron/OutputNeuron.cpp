#include "OutputNeuron.h"

namespace busybin {
  /**
   * Init.
   */
  OutputNeuron::OutputNeuron() : ideal(0) {
  }

  /**
   * Nothing to connect to.
   */
  void OutputNeuron::connectTo(Neuron&, double weight) {
  }

  /**
   * Nothing to feed forward to.
   */
  void OutputNeuron::feedForward() const {
  }

  /**
   * Set the ideal output for this Neuron.
   */
  void OutputNeuron::setIdeal(double ideal) {
    this->ideal = ideal;
  }

  /**
   * Get the ideal output for this Neuron.
   */
  double OutputNeuron::getIdeal() const {
    return this->ideal;
  }

  /**
   * Update the weights on inbound connections given the expected output.
   */
  void OutputNeuron::updateErrorTerm() {
    // See the note in Neuron's implementation for derivation references.
    // OutputNeurons have a specialized (simplified) calculation.
    double out = this->getOutput();

    this->errorTerm = (out - this->ideal) * out * (1 - out);
  }

  /**
   * Nothing to update (no forward-connected Neurons).
   */
  void OutputNeuron::updateWeights(double learningRate) {
  }

  /**
   * Get the name.
   */
  string OutputNeuron::getName() const {
    return "OutputNeuron";
  }

  /**
   * Same as the other Neurons, but with the ideal.
   */
  string OutputNeuron::toString() const {
    ostringstream oss;

    oss << Neuron::toString() << ' '
        << "Ideal: "          << this->getIdeal();

    return oss.str();
  }
}

