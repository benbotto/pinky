#include "Neuron.h"

namespace busybin {
  /**
   * Init.  Output and net input default to 0.
   */
  Neuron::Neuron() : netInput(0), output(0) {
  }

  /**
   * Do-nothing destructor.
   */
  Neuron::~Neuron() {
  }

  /**
   * Add a connection to a Neuron in a forward layer (in front of this one).
   */
  void Neuron::connectTo(Neuron& neuron, double weight) {
    this->connections.push_back(make_pair(&neuron, weight));
  }

  /**
   * Get the net input for this Neuron.
   */
  double Neuron::getNetInput() const {
    return this->netInput;
  }

  /**
   * Get the output of this Neuron.
   */
  double Neuron::getOutput() const {
    return this->output;
  }

  /**
   * Add an input-weight pair from a Neuron.  Note that the weight here is
   * inbound (e.g. on the inward connection) as opposed to the connection
   * weights.
   */
  void Neuron::pushInput(double input, double weight) {
    this->inputs.push_back(input);
    this->weights.push_back(weight);
  }

  /**
   * Push output from this neuron into its forward-connected neurons.
   */
  void Neuron::feedForward() const {
    for (const connection_t& conn: this->connections) {
      conn.first->pushInput(this->getOutput(), conn.second);
    }
  }

  /**
   * Clear the inputs and weights that were pushed from backward connections
   * in the last feed-forward iteration.
   */
  void Neuron::reset() {
    this->inputs.clear();
    this->weights.clear();
  }

  /**
   * Update the output based on the input values and weights.
   */
  void Neuron::updateOutput() {
    // Dot product of inputs and weights.  Note that the initial value (0.0)
    // needs to be a double.  0 will not work.
    this->netInput = inner_product(
      this->inputs.begin(), this->inputs.end(),
      this->weights.begin(), 0.0);
    
    // Squash the output using the logistic function.
    this->output = 1 / (1 + exp(-this->netInput));
  }

  /**
   * Get the name.
   */
  string Neuron::getName() const {
    return "Neuron";
  }

  /**
   * Describe this Neuron.
   */
  string Neuron::toString() const {
    ostringstream oss;

    oss << "Name: "      << this->getName()     << ' '
        << "Net Input: " << this->getNetInput() << ' '
        << "Output: "    << this->getOutput();

    return oss.str();
  }

  /**
   * Output the Neuron to the stream os.
   */
  ostream& operator<<(ostream& os, const Neuron& neuron) {  
    os << neuron.toString();

    return os;
  }
}

