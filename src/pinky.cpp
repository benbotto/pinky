#include <iostream>
using std::cout;
using std::endl;

#include "NeuralNet.h"
#include "neuron/Neuron.h"
#include "neuron/InputNeuron.h"
#include "neuron/BiasNeuron.h"

int main(int argc, char* argv[]) {
  busybin::NeuralNet<2, 2, 2> nn;

  cout << nn << endl;

  for (unsigned i = 0; i < 50000000; ++i) {
    nn.train({.05, .10}, {.01, .99});

    if (i % 1000000 == 0) {
      cout << "Iteration: " << i << '\n'
           << nn << endl;
    }
  }

  return 0;
}

