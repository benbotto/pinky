#ifndef _BUSYBIN_OUTPUT_NEURON_H_
#define _BUSYBIN_OUTPUT_NEURON_H_

#include "Neuron.h"

namespace busybin {
  class OutputNeuron : public Neuron {
    double ideal;

  public:
    OutputNeuron();
    void connectTo(Neuron&, double weight);
    void feedForward() const;
    void setIdeal(double ideal);
    double getIdeal() const;
    void updateErrorTerm();
    void updateWeights(double learningRate);
    string getName() const;
    string toString() const;
  };
}

#endif

