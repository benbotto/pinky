#ifndef _BUSYBIN_NEURAL_NET_H_
#define _BUSYBIN_NEURAL_NET_H_

#include <array>
using std::array;
#include <memory>
using std::unique_ptr;
#include <iostream>
using std::cout;
using std::endl;
#include "neuron/Neuron.h"
#include "neuron/InputNeuron.h"
#include "neuron/BiasNeuron.h"
#include "neuron/OutputNeuron.h"
using busybin::Neuron;
using busybin::InputNeuron;
using busybin::BiasNeuron;
using busybin::OutputNeuron;

namespace busybin {
  // Sizes do _not_ include the bias nodes.  A bias node is
  // added automatically for the input and hidden layers.
  template <size_t NUM_IN, size_t NUM_HIDDEN, size_t NUM_OUT>
  class NeuralNet {
    typedef unique_ptr<Neuron> pNeuron;

    array<pNeuron, NUM_IN + 1>     inputLayer;
    array<pNeuron, NUM_HIDDEN + 1> hiddenLayer;
    array<pNeuron, NUM_OUT>        outputLayer;

  public:
    /**
     * Initialize the network.
     */
    NeuralNet() {
      // Input layer, with a bias at the end.
      for (unsigned i = 0; i < NUM_IN; ++i)
        this->inputLayer[i] = pNeuron(new InputNeuron());
      this->inputLayer[NUM_IN] = pNeuron(new BiasNeuron());

      // Hidden layer, with a bias at the end.
      for (unsigned i = 0; i < NUM_HIDDEN; ++i)
        this->hiddenLayer[i] = pNeuron(new Neuron());
      this->hiddenLayer[NUM_HIDDEN] = pNeuron(new BiasNeuron());

      // Output layer.
      for (unsigned i = 0; i < NUM_OUT; ++i)
        this->outputLayer[i] = pNeuron(new OutputNeuron());

      // Each input neuron gets connected to each hidden neuron.
      array<double, 6> iWeights = {.15, .25, .20, .30, .35, .35};
      for (unsigned i = 0; i < NUM_IN + 1; ++i) {
        for (unsigned h = 0; h < NUM_HIDDEN; ++h) {
          // Connection is held by the backward neuron, so here the input
          // neuron connects forward into the hidden neuron.
          cout << "Connection from input " << i
               << " to hidden " << h
               << " with weight " << iWeights[NUM_IN * i + h]
               << endl;
          this->inputLayer[i]->connectTo(*this->hiddenLayer[h], iWeights[NUM_IN * i + h]);
        }
      }

      // Each hidden neuron gets connected to each output neuron.
      array<double, 6> hWeights = {.40, .50, .45, .55, .60, .60};
      for (unsigned h = 0; h < NUM_HIDDEN + 1; ++h) {
        for (unsigned o = 0; o < NUM_OUT; ++o) {
          // Hidden neuron connects forward to output neuron.
          cout << "Connection from hidden " << h
               << " to output " << o
               << " with weight " << hWeights[NUM_HIDDEN * h + o]
               << endl;
          this->hiddenLayer[h]->connectTo(*this->outputLayer[o], hWeights[NUM_HIDDEN * h + o]);
        }
      }
    }

    /**
     * Do a round of training.
     */
    void train(const array<double, NUM_IN>& inputs) const {
      // Reset the inputs/weights from the last round.  This is only needed
      // on the hidden and output layers, as the inputs and biases don't have
      // any connections.
      for (const pNeuron& pNeuron: this->hiddenLayer)
        pNeuron->reset();

      for (const pNeuron& pNeuron: this->outputLayer)
        pNeuron->reset();

      // Set the new inputs.
      for (unsigned i = 0; i < NUM_IN; ++i)
        this->inputLayer[i]->pushInput(inputs[i]);

      // Feed the inputs forward to the hidden layer.
      for (unsigned i = 0; i < NUM_IN + 1; ++i)
        this->inputLayer[i]->feedForward();

      // Update the outputs for each hidden neuron.
      for (unsigned i = 0; i < NUM_HIDDEN; ++i)
        this->hiddenLayer[i]->updateOutput();

      // Feed the hidden outputs forward to the output layer.
      for (unsigned i = 0; i < NUM_HIDDEN + 1; ++i)
        this->hiddenLayer[i]->feedForward();

      // Finally, update the outputs.
      for (unsigned i = 0; i < NUM_OUT; ++i)
        this->outputLayer[i]->updateOutput();

      for (unsigned i = 0; i < NUM_IN + 1; ++i) {
        cout << "Input neuron " << i
             << " output: "     << this->inputLayer[i]->getOutput() << endl;
      }

      for (unsigned i = 0; i < NUM_HIDDEN + 1; ++i) {
        cout << "Hidden neuron " << i
             << " output: "     << this->hiddenLayer[i]->getOutput() << endl;
      }

      for (unsigned i = 0; i < NUM_OUT; ++i) {
        cout << "Output neuron " << i
             << " output: "     << this->outputLayer[i]->getOutput() << endl;
      }
    }
  };
}

#endif

