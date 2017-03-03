#ifndef _BUSYBIN_NEURON_H_
#define _BUSYBIN_NEURON_H_

#include <utility>
using std::pair;
using std::make_pair;
#include <vector>
using std::vector;
#include <numeric>
using std::inner_product;
#include <string>
using std::string;
#include <cmath>

namespace busybin {
  class Neuron {
    typedef pair<Neuron*, double> connection_t;

  protected:
    double output;
    vector<connection_t> connections;
    vector<double>       inputs;
    vector<double>       weights;

    // Push an input from a Neuron in a backward layer.
    virtual void pushInput(double input, double weight = 1);
  public:
    Neuron();
    virtual ~Neuron();

    // Connect this Neuron to a Neuron that is in a forward layer.
    virtual void connectTo(Neuron&, double weight);

    // Get this Neuron's output.
    virtual double getOutput() const;

    // Fire the output across the synapses (e.g. push output to connected
    // Neurons).
    virtual void feedForward() const;

    // Clear inputs and weights.
    virtual void reset();

    // Update this Neuron's output value.
    virtual void updateOutput();

    // Get the name (type) of this neuron.
    virtual string getName() const;
  };
}

#endif

