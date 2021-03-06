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
    // after a forward pass during a training run.
    double totalError;

    // This is the outputs of the system, which is set after a forward pass.
    array<double, NUM_OUT> outputs;

    /**
     * Initialize the network.  This is called from the various overloaded
     * ctors, and does the actual initialization of the network.
     */
    void initialize(const array<double, (NUM_IN + 1) * NUM_HIDDEN>& iWeights,
      const array<double, (NUM_HIDDEN + 1) * NUM_OUT>& hWeights) {
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
          this->pInputLayer[i]->connectTo(*this->pHiddenLayer[h], iWeights[NUM_HIDDEN * i + h]);
        }
      }

      // Each hidden neuron gets connected to each output neuron.
      for (unsigned h = 0; h < NUM_HIDDEN + 1; ++h) {
        for (unsigned o = 0; o < NUM_OUT; ++o) {
          // Hidden neuron connects forward to output neuron.
          this->pHiddenLayer[h]->connectTo(*this->pOutputLayer[o], hWeights[NUM_OUT * h + o]);
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
     * Initialize the network with input and hidden weights.
     */
    NeuralNet(array<double, (NUM_IN + 1) * NUM_HIDDEN> iWeights,
      array<double, (NUM_HIDDEN + 1) * NUM_OUT> hWeights) {
      this->initialize(iWeights, hWeights);
    }

    /**
     * Initialize the network with a single array of weights.
     */
    NeuralNet(array<double, (NUM_IN + 1) * NUM_HIDDEN + (NUM_HIDDEN + 1) * NUM_OUT> weights) {
      array<double, (NUM_IN     + 1) * NUM_HIDDEN> iWeights;
      array<double, (NUM_HIDDEN + 1) * NUM_OUT>    hWeights;

      for (unsigned i = 0; i < iWeights.size(); ++i)
        iWeights[i] = weights[i];
      for (unsigned i = 0; i < hWeights.size(); ++i)
        hWeights[i] = weights[iWeights.size() + i];

      this->initialize(iWeights, hWeights);
    }

    /**
     * Do a round of training.
     */
    double train(const array<double, NUM_IN>& inputs,
      const array<double, NUM_OUT>& expected, double learningRate = .5) {

      /**
       * Forward pass.
       */

      this->feedForward(inputs);

      // Calculate the total error.
      this->totalError = 0;

      // E_{total} = \sum \frac{1}{2}(target - output)^{2}
      // Note that the 1/2 is there so that the exponent cancels out when
      // the derivative is taken.  A learning rate will be used, so the
      // constant won't matter in the long run.
      for (unsigned i = 0; i < NUM_OUT; ++i)
        this->totalError += std::pow(expected[i] - this->outputs[i], 2);
      this->totalError *= .5;

      /**
       * Backward pass.
       */

      // Set the ideal outputs.
      for (unsigned i = 0; i < NUM_OUT; ++i)
        dynamic_cast<OutputNeuron&>(*this->pOutputLayer[i]).setIdeal(expected[i]);

      // Compute the error term for the output neurons.
      for (unsigned i = 0; i < NUM_OUT; ++i)
        this->pOutputLayer[i]->updateErrorTerm();

      // Compute the error term for the hidden neurons, which rely on the
      // error terms of the output neurons.
      for (unsigned i = 0; i < NUM_HIDDEN; ++i)
        this->pHiddenLayer[i]->updateErrorTerm();

      // Update the weights between the hidden layer and the output layer.
      for (unsigned i = 0; i < NUM_HIDDEN + 1; ++i)
        this->pHiddenLayer[i]->updateWeights(learningRate);

      // Update the weights between the input layer and the hidden.
      for (unsigned i = 0; i < NUM_IN + 1; ++i)
        this->pInputLayer[i]->updateWeights(learningRate);

      return this->getTotalError();
    }

    /**
     * Feed the inputs forward and return the outputs.
     */
    array<double, NUM_OUT> feedForward(const array<double, NUM_IN>& inputs) {
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

      // Store the new outputs.
      for (unsigned i = 0; i < NUM_OUT; ++i)
        this->outputs[i] = this->pOutputLayer[i]->getOutput();

      return this->getOutputs();
    }

    /**
     * Get the total error for the system.  Computed after a forward pass.
     */
    double getTotalError() const {
      return this->totalError;
    }

    /**
     * Get the outputs for the system (available after a forward pass).
     */
    array<double, NUM_OUT> getOutputs() const {
      return this->outputs;
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
     * Get the weights for all Neurons in the network, left to right, top to
     * bottom.
     */
    array<double, (NUM_IN + 1) * NUM_HIDDEN + (NUM_HIDDEN + 1) * NUM_OUT> getWeights() const {
      array<double, (NUM_IN + 1) * NUM_HIDDEN + (NUM_HIDDEN + 1) * NUM_OUT> weights;
      unsigned w = 0;

      for (const pNeuron_t& pNeuron : this->neurons) {
        vector<double> nWeights = pNeuron->getWeights();

        for (double weight : nWeights)
          weights[w++] = weight;
      }

      return weights;
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

