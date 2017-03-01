#include <iostream>
using std::cout;
using std::endl;

#include "NeuralNet.h"
#include "neuron/Neuron.h"
#include "neuron/InputNeuron.h"
#include "neuron/BiasNeuron.h"

int main(int argc, char* argv[]) {
  busybin::NeuralNet<2, 2, 2> nn;

  nn.train({.05, .10});

  return 0;
}

