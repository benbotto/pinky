#ifndef _BUSYBIN_NEURAL_NET_H_
#define _BUSYBIN_NEURAL_NET_H_

#include <array>
using std::array;
#include <memory>
using std::unique_ptr;
#include <sstream>
using std::ostringstream;
#include <ostream>
using std::ostream;
#include <vector>
using std::vector;
#include <random>
using std::uniform_real_distribution;
using std::default_random_engine;
using std::random_device;
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

    // This is the total error of the system, which is calculated
    // after a forward pass.
    double totalError;

    /**
     * Initialize the network.  This is called from the various overloaded
     * ctors, and does the actual initialization of the network.
     */
    void initialize(array<double, (NUM_IN + 1) * NUM_HIDDEN> iWeights,
      array<double, (NUM_HIDDEN + 1) * NUM_OUT> hWeights) {
      this->pInputLayer  = &neurons[0];
      this->pHiddenLayer = &neurons[NUM_IN + 1];
      this->pOutputLayer = &neurons[NUM_IN + 1 + NUM_HIDDEN + 1];
      this->totalError   = 0;

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
      for (unsigned i = 0; i < NUM_IN + 1; ++i) {
        for (unsigned h = 0; h < NUM_HIDDEN; ++h) {
          // Connection is held by the backward neuron, so here the input
          // neuron connects forward into the hidden neuron.
          this->pInputLayer[i]->connectTo(*this->pHiddenLayer[h], iWeights[NUM_IN * i + h]);
        }
      }

      // Each hidden neuron gets connected to each output neuron.
      for (unsigned h = 0; h < NUM_HIDDEN + 1; ++h) {
        for (unsigned o = 0; o < NUM_OUT; ++o) {
          // Hidden neuron connects forward to output neuron.
          this->pHiddenLayer[h]->connectTo(*this->pOutputLayer[o], hWeights[NUM_HIDDEN * h + o]);
        }
      }
    }

  public:
    /**
     * Initialize the network with random weights.
     */
    NeuralNet() {
      array<double, (NUM_IN     + 1) * NUM_HIDDEN> iWeights;
      array<double, (NUM_HIDDEN + 1) * NUM_OUT>    hWeights;
      uniform_real_distribution<double>            dist(-1, 1);
      random_device                                randDev;
      default_random_engine                        engine(randDev());

      for (unsigned i = 0; i < iWeights.size(); ++i)
        iWeights[i] = dist(engine);
      for (unsigned h = 0; h < hWeights.size(); ++h)
        hWeights[h] = dist(engine);

      this->initialize(iWeights, hWeights);
    }

    /**
     * Initialize the network with a given set of weights.
     */
    NeuralNet(array<double, (NUM_IN + 1) * NUM_HIDDEN> iWeights,
      array<double, (NUM_HIDDEN + 1) * NUM_OUT> hWeights) {
      this->initialize(iWeights, hWeights);
    }

    /**
     * Do a round of training.
     */
    void train(const array<double, NUM_IN>& inputs,
      const array<double, NUM_OUT>& expected) {

      /**
       * Forward pass.
       */

      // Reset the inputs/weights from the last training round.
      for (unsigned i = NUM_IN + 1; i < this->neurons.size(); ++i)
        this->neurons[i]->reset();

      // Set the new inputs.
      for (unsigned i = 0; i < NUM_IN; ++i)
        dynamic_cast<InputNeuron&>(*this->pInputLayer[i]).pushInput(inputs[i]);

      // Set the ideal outputs.
      for (unsigned i = 0; i < NUM_OUT; ++i)
        dynamic_cast<OutputNeuron&>(*this->pOutputLayer[i]).setIdeal(expected[i]);

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

      // Calculate the total error.
      this->totalError = 0;

      for (unsigned i = 0; i < NUM_OUT; ++i) {
        // E_{total} = \sum \frac{1}{2}(target - output)^{2}
        // Note that the 1/2 is there so that the exponent cancels out when
        // the derivative is taken.  A learning rate will be used, so the
        // constant won't matter in the long run.
        this->totalError += .5 * std::pow(expected[i] - this->pOutputLayer[i]->getOutput(), 2);
      }

      /**
       * Backward pass.
       */

      // Compute the error term for the output neurons.
      for (unsigned i = 0; i < NUM_OUT; ++i)
        this->pOutputLayer[i]->updateErrorTerm();

      // Compute the error term for the hidden neurons, which rely on the
      // error terms of the output neurons.
      for (unsigned i = 0; i < NUM_HIDDEN; ++i)
        this->pHiddenLayer[i]->updateErrorTerm();

      // Update the weights between the hidden layer and the output layer.
      for (unsigned i = 0; i < NUM_HIDDEN + 1; ++i)
        this->pHiddenLayer[i]->updateWeights();

      // Update the weights between the input layer and the hidden.
      for (unsigned i = 0; i < NUM_IN + 1; ++i)
        this->pInputLayer[i]->updateWeights();
    }

    /**
     * Get the total error for the system.  Computed after a forward pass.
     */
    double getTotalError() const {
      return this->totalError;
    }

    /**
     * Describe the network.
     */
    string toString() const {
      ostringstream  oss;
      vector<double> weights;

      oss << "Total error: " << this->getTotalError() << '\n';

      // Print all the neurons.
      for (const pNeuron_t& pNeuron: this->neurons)
        oss << *pNeuron << '\n';
      oss << '\n';

      // Input to hidden weights.
      for (unsigned i = 0; i < NUM_IN; ++i) {
        weights = this->pInputLayer[i]->getWeights();

        for (unsigned h = 0; h < NUM_HIDDEN; ++h)
          oss << "\tWeight_I" << i << ",H" << h << ": " << weights[h] << '\n';
      }

      // First bias weight.
      weights = this->pInputLayer[NUM_IN]->getWeights();
      for (unsigned h = 0; h < NUM_HIDDEN; ++h)
        oss << "\tWeight_B1" << ",H" << h << ": " << weights[h] << '\n';

      // Hidden to output weights.
      oss << '\n';
      for (unsigned h = 0; h < NUM_HIDDEN; ++h) {
        weights = this->pHiddenLayer[h]->getWeights();

        for (unsigned o = 0; o < NUM_OUT; ++o)
          oss << "\tWeight_H" << h << ",O" << o << ": " << weights[o] << '\n';
      }

      // Second bias weight.
      weights = this->pHiddenLayer[NUM_HIDDEN]->getWeights();
      for (unsigned o = 0; o < NUM_OUT; ++o)
        oss << "\tWeight_B1" << ",O" << o << ": " << weights[o] << '\n';

      return oss.str();
    }

    /**
     * Print the network to a stream.
     */
    friend ostream& operator<<(ostream& os, const NeuralNet& net) {
      os << net.toString();

      return os;
    }
  };
}

#endif

