#include "NeuralNet.h"

#include <iostream>
using std::cout;
using std::endl;

namespace busybin {
  NeuralNet::NeuralNet(unsigned numIn, unsigned numHidden, unsigned numOut) :
    numIn(numIn), numHidden(numHidden), numOut(numOut) {

    // Input layer, with a bias at the end.
    for (unsigned i = 0; i < numIn + 1; ++i)
      layers[0].push_back(unique_ptr<Neuron>(new Neuron()));

    // Hidden layer, with a bias at the end.
    for (unsigned i = 0; i < numHidden + 1; ++i)
      layers[1].push_back(unique_ptr<Neuron>(new Neuron()));

    // Output layer.
    for (unsigned i = 0; i < numOut; ++i)
      layers[2].push_back(unique_ptr<Neuron>(new Neuron()));

    // Each input neuron gets connected to each hidden neuron.
    array<double, 6> hWeights = {.15, .20, .25, .30, .35, .35};
    for (unsigned i = 0; i < numIn + 1; ++i) {
      for (unsigned h = 0; h < numHidden; ++h) {
        // Connection is held by the forward neuron, so here the hidden
        // neuron receives the connection from the input neuron.
        cout << "Connection from input " << i
             << " to hidden " << h
             << " with weight " << hWeights[numIn * i + h]
             << endl;
        layers[1].at(h)->addInput(*layers[0].at(i), hWeights[numIn * i + h]);
      }
    }

    // Each hidden neuron gets connected to each output neuron.
    array<double, 6> oWeights = {.40, .45, .50, .55, .60, .60};
    for (unsigned h = 0; h < numHidden + 1; ++h) {
      for (unsigned o = 0; o < numOut; ++o) {
        // Output neuron receives the connection from the hidden neuron.
        cout << "Connection from hidden " << h
             << " to output " << o
             << " with weight " << oWeights[numHidden * h + o]
             << endl;
        layers[2].at(o)->addInput(*layers[1].at(h), oWeights[numHidden * h + o]);
      }
    }
  }
}

