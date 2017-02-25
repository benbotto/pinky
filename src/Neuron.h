#ifndef _BUSYBIN_NEURON_H_
#define _BUSYBIN_NEURON_H_

#include <vector>
using std::vector;
#include <utility>
using std::pair;

namespace busybin {
  class Neuron {
    vector<pair<const Neuron* const, double> > inputs;

  public:
    Neuron& addInput(const Neuron& neuron, double weight);
  };
}

#endif

