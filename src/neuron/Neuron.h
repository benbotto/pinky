#ifndef _BUSYBIN_NEURON_H_
#define _BUSYBIN_NEURON_H_

#include <vector>
using std::vector;
#include <utility>
using std::pair;
#include "INeuron.h"

namespace busybin {
  class Neuron : public INeuron {
    vector<pair<INeuron* const, double> > inputs;

  public:
    void addInput(INeuron& neuron, double weight);
    double getOutput() const;
  };
}

#endif

