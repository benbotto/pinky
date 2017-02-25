#ifndef _BUSYBIN_NEURAL_NET_H_
#define _BUSYBIN_NEURAL_NET_H_

#include <vector>
using std::vector;
#include <array>
using std::array;
#include <memory>
using std::unique_ptr;
#include "Neuron.h"
using busybin::Neuron;

namespace busybin {
  class NeuralNet {
    typedef unique_ptr<Neuron> pNeuron;
    typedef vector<pNeuron>    layer;

    unsigned numIn, numHidden, numOut;

    array<layer, 3> layers;

  public:
    NeuralNet(unsigned numInputs, unsigned numHidden, unsigned numOutputs);
  };
}


#endif

