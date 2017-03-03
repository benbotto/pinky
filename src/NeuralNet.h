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
    typedef unique_ptr<Neuron> pNeuron_t;

    // All the neurons in the network.
    array<pNeuron_t, NUM_IN + 1 + NUM_HIDDEN + 1 + NUM_OUT> neurons;

    // These are convenience pointers that point to each layer.  Keep in mind
    // that the input and hidden layer each have an additional bias neuron.
    pNeuron_t* pInputLayer;
    pNeuron_t* pHiddenLayer;
    pNeuron_t* pOutputLayer;

  public:
    /**
     * Initialize the network.
     */
    NeuralNet() {
      this->pInputLayer  = &neurons[0];
      this->pHiddenLayer = &neurons[NUM_IN + 1];
      this->pOutputLayer = &neurons[NUM_IN + 1 + NUM_HIDDEN + 1];

      // Input layer, with a bias at the end.
      for (unsigned i = 0; i < NUM_IN; ++i)
        this->pInputLayer[i] = pNeuron_t(new InputNeuron());
      this->pInputLayer[NUM_IN] = pNeuron_t(new BiasNeuron());

      // Hidden layer, with a bias at the end.
      for (unsigned i = 0; i < NUM_HIDDEN; ++i)
        this->pHiddenLayer[i] = pNeuron_t(new Neuron());
      this->pHiddenLayer[NUM_HIDDEN] = pNeuron_t(new BiasNeuron());

      // Output layer.
      for (unsigned i = 0; i < NUM_OUT; ++i)
        this->pOutputLayer[i] = pNeuron_t(new OutputNeuron());

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
          this->pInputLayer[i]->connectTo(*this->pHiddenLayer[h], iWeights[NUM_IN * i + h]);
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
          this->pHiddenLayer[h]->connectTo(*this->pOutputLayer[o], hWeights[NUM_HIDDEN * h + o]);
        }
      }
    }

    /**
     * Do a round of training.
     */
    void train(const array<double, NUM_IN>& inputs,
      const array<double, NUM_OUT>& expected) const {
      double totalError = 0;

      // Reset the inputs/weights from the last training round.
      for (unsigned i = NUM_IN + 1; i < this->neurons.size(); ++i)
        this->neurons[i]->reset();

      // Set the new inputs.
      for (unsigned i = 0; i < NUM_IN; ++i)
        dynamic_cast<InputNeuron&>(*this->pInputLayer[i]).pushInput(inputs[i]);

      // Feed the inputs forward to the hidden layer.
      for (unsigned i = 0; i < NUM_IN + 1; ++i)
        this->pInputLayer[i]->feedForward();

      // Update the outputs for each hidden neuron.
      for (unsigned i = 0; i < NUM_HIDDEN; ++i)
        this->pHiddenLayer[i]->updateOutput();

      // Feed the hidden outputs forward to the output layer.
      for (unsigned i = 0; i < NUM_HIDDEN + 1; ++i)
        this->pHiddenLayer[i]->feedForward();

      // Finally, update the outputs.
      for (unsigned i = 0; i < NUM_OUT; ++i)
        this->pOutputLayer[i]->updateOutput();

      for (const pNeuron_t& pNeuron: this->neurons) {
        cout << *pNeuron << endl;
      }

      // Calculate the total error.
      for (unsigned i = 0; i < NUM_OUT; ++i) {
        // E_{total} = \sum \frac{1}{2}(target - output)^{2}
        // Note that the 1/2 is there so that the exponent cancels out when
        // the derivative is taken.  A learning rate will be used, so the
        // constant won't matter in the long run.
        totalError += .5 * std::pow(expected[i] - this->pOutputLayer[i]->getOutput(), 2);
      }

      cout << "Total error: " << totalError << endl;
    }
  };
}

#endif

