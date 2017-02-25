#ifndef _BUSYBIN_NEURAL_NET_H_
#define _BUSYBIN_NEURAL_NET_H_

#include <vector>
using std::vector;
#include <array>
using std::array;
#include <memory>
using std::unique_ptr;
#include "neuron/INeuron.h"
#include "neuron/Neuron.h"
#include "neuron/InputNeuron.h"
#include "neuron/BiasNeuron.h"
using busybin::INeuron;
using busybin::InputNeuron;
using busybin::BiasNeuron;

namespace busybin {
  class NeuralNet {
    typedef unique_ptr<INeuron> pNeuron;
    typedef vector<pNeuron>     layer;

    unsigned numIn, numHidden, numOut;

    array<layer, 3> layers;

  public:
    NeuralNet(unsigned numInputs, unsigned numHidden, unsigned numOutputs);
  };
}


#endif

