#include <iostream>
using std::cout;
using std::endl;
#include <array>
using std::array;
#include <exception>
using std::exception;
#include "NeuralNet.h"
#include "NeuralNetMomento.h"
#include "neuron/Neuron.h"
#include "neuron/InputNeuron.h"
#include "neuron/BiasNeuron.h"

int main(int argc, char* argv[]) {
  try {
    busybin::NeuralNetMomento<2, 2, 2> momento;
    busybin::NeuralNet<2, 2, 2>        nn = momento.load("weights.bin");;

    cout << nn << endl;
    nn.train({.05, .10}, {.01, .99});
    cout << nn << endl;

    for (double weight : nn.getWeights())
      cout << weight << endl;
  }
  catch (exception& ex) {
    cout << "Error: " << ex.what() << endl;
  }

  return 0;
}

