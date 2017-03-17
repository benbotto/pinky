#include <iostream>
using std::cout;
using std::endl;
#include <array>
using std::array;

#include "NeuralNet.h"
#include "neuron/Neuron.h"
#include "neuron/InputNeuron.h"
#include "neuron/BiasNeuron.h"

int main(int argc, char* argv[]) {
  //array<double, 6> iWeights = {.15, .25, .20, .30, .35, .35};
  //array<double, 6> hWeights = {.40, .50, .45, .55, .60, .60};
  //busybin::NeuralNet<2, 2, 2> nn(iWeights, hWeights);
  busybin::NeuralNet<2, 2, 2> nn;

  cout << nn << endl;
  nn.train({.05, .10}, {.01, .99});
  cout << nn << endl;

  /*for (unsigned i = 0; i < 50000000; ++i) {
    nn.train({.05, .10}, {.01, .99});

    if (i % 1000000 == 0) {
      cout << "Iteration: " << i << '\n'
           << nn << endl;
    }
  }*/

  return 0;
}

